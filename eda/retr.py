"""Memory-light sparse retrieval: integer features + inverted index over Source 1 + numba top-K.

Strings must already be normalised to [a-z0-9 ] (see norm.basic). Feature modes:
  0 char3   - exact trigram ids (base-37) over ' ' + s + ' ', runs of spaces collapsed
  1 char3ns - trigrams over the string with spaces removed ('mim brothers' == 'mimbrothers')
  2 word    - FNV-1a hashed tokens (len>=2) into 2**22 buckets
Cost of one query = sum of posting-list lengths of its (non-capped) features.
"""
import numpy as np
import numba as nb

NFEAT = {0: 37 ** 3, 1: 37 ** 3, 2: 1 << 22}
MAXF = 1024  # max features per row (names/addresses are far shorter)


def to_buf(strings):
    """iterable of ascii strings -> (uint8 buffer, int64 offsets)."""
    strings = [s or "" for s in strings]
    lens = np.fromiter((len(s) for s in strings), dtype=np.int64, count=len(strings))
    off = np.zeros(len(strings) + 1, dtype=np.int64); np.cumsum(lens, out=off[1:])
    buf = np.frombuffer("".join(strings).encode("ascii", "replace"), dtype=np.uint8)
    return buf, off


@nb.njit(cache=True)
def _code(c):
    if 97 <= c <= 122: return c - 96           # a-z -> 1..26
    if 48 <= c <= 57: return c - 21            # 0-9 -> 27..36
    return 0                                     # space / other


@nb.njit(cache=True)
def _row_feats(buf, a, b, mode, tmp):
    k = 0
    if mode == 2:
        h = np.uint32(2166136261); L = 0
        for i in range(a, b + 1):
            c = buf[i] if i < b else 32
            if c == 32:
                if L >= 2 and k < MAXF:
                    tmp[k] = np.int32(h & np.uint32((1 << 22) - 1)); k += 1
                h = np.uint32(2166136261); L = 0
            else:
                h = (h ^ np.uint32(c)) * np.uint32(16777619); L += 1
        return k
    # char trigrams over a cleaned code sequence
    seq = np.empty(b - a + 2, dtype=np.int32); m = 0
    if mode == 0:
        seq[m] = 0; m += 1
    for i in range(a, b):
        x = _code(buf[i])
        if x == 0:
            if mode == 1 or seq[m - 1] == 0:
                continue
        seq[m] = x; m += 1
    if mode == 0:
        if seq[m - 1] != 0:
            seq[m] = 0; m += 1
    for i in range(m - 2):
        if k < MAXF:
            tmp[k] = seq[i] * 1369 + seq[i + 1] * 37 + seq[i + 2]; k += 1
    return k


@nb.njit(cache=True)
def featurize(buf, off, mode):
    """Returns CSR (indptr, ids): sorted unique feature ids per row."""
    n = len(off) - 1
    ids = np.empty(len(buf) + 3 * n + 1, dtype=np.int32)
    indptr = np.zeros(n + 1, dtype=np.int64)
    tmp = np.empty(MAXF, dtype=np.int32)
    pos = 0
    for r in range(n):
        k = _row_feats(buf, off[r], off[r + 1], mode, tmp)
        if k > 0:
            s = np.sort(tmp[:k])
            ids[pos] = s[0]; pos += 1
            for j in range(1, k):
                if s[j] != s[j - 1]:
                    ids[pos] = s[j]; pos += 1
        indptr[r + 1] = pos
    return indptr, ids[:pos].copy()


@nb.njit(cache=True)
def build_index(indptr, ids, nfeat):
    """Inverted index over docs. Returns df, idf, post_ptr, post_doc, doc_norm."""
    n = len(indptr) - 1
    df = np.zeros(nfeat, dtype=np.int32)
    for i in range(len(ids)):
        df[ids[i]] += 1
    idf = np.zeros(nfeat, dtype=np.float32)
    for f in range(nfeat):
        if df[f] > 0:
            idf[f] = np.log((n + 1.0) / df[f])
    post_ptr = np.zeros(nfeat + 1, dtype=np.int64)
    for f in range(nfeat):
        post_ptr[f + 1] = post_ptr[f] + df[f]
    fill = post_ptr[:-1].copy()
    post_doc = np.empty(len(ids), dtype=np.int32)
    doc_norm = np.zeros(n, dtype=np.float32)
    for d in range(n):
        s = 0.0
        for j in range(indptr[d], indptr[d + 1]):
            f = ids[j]
            post_doc[fill[f]] = d; fill[f] += 1
            s += idf[f] * idf[f]
        doc_norm[d] = np.sqrt(s) if s > 0 else 1.0
    return df, idf, post_ptr, post_doc, doc_norm


@nb.njit(cache=True)
def _heap_push(hs, hi, size, K, s, d):
    # min-heap of size K on score
    if size < K:
        i = size; size += 1
        hs[i] = s; hi[i] = d
        while i > 0:
            p = (i - 1) // 2
            if hs[p] <= hs[i]: break
            hs[p], hs[i] = hs[i], hs[p]; hi[p], hi[i] = hi[i], hi[p]; i = p
    elif s > hs[0]:
        hs[0] = s; hi[0] = d; i = 0
        while True:
            l = 2 * i + 1; r = l + 1; m = i
            if l < K and hs[l] < hs[m]: m = l
            if r < K and hs[r] < hs[m]: m = r
            if m == i: break
            hs[m], hs[i] = hs[i], hs[m]; hi[m], hi[i] = hi[i], hi[m]; i = m
    return size


@nb.njit(cache=True)
def _query_block(q0, q1, qptr, qids, idf, df, post_ptr, post_doc, doc_norm, max_df, min_keep, K, out_d, out_s, work):
    """Features processed rarest-first; a feature with df > max_df is skipped unless fewer than
    min_keep features have been processed so far (so every query gets its min_keep rarest features)."""
    ndoc = len(doc_norm)
    acc = np.zeros(ndoc, dtype=np.float32)
    touched = np.empty(ndoc, dtype=np.int32)
    hs = np.empty(K, dtype=np.float32); hi = np.empty(K, dtype=np.int32)
    fdf = np.empty(MAXF, dtype=np.int64); fid = np.empty(MAXF, dtype=np.int32)
    for q in range(q0, q1):
        nt = 0; qn = 0.0; w = 0; m = 0
        for j in range(qptr[q], qptr[q + 1]):
            f = qids[j]
            if df[f] == 0: continue
            qn += idf[f] * idf[f]
            fdf[m] = df[f]; fid[m] = f; m += 1
        order = np.argsort(fdf[:m])
        used = 0
        for oi in range(m):
            f = fid[order[oi]]
            if df[f] > max_df and used >= min_keep: break
            used += 1
            wt = idf[f] * idf[f]
            for p in range(post_ptr[f], post_ptr[f + 1]):
                d = post_doc[p]
                if acc[d] == 0.0:
                    touched[nt] = d; nt += 1
                acc[d] += wt
            w += df[f]
        work[q] = w
        size = 0
        qn = np.sqrt(qn) if qn > 0 else 1.0
        for t in range(nt):
            d = touched[t]
            size = _heap_push(hs, hi, size, K, acc[d] / (qn * doc_norm[d]), d)
            acc[d] = 0.0
        srt = np.argsort(-hs[:size])
        for t in range(size):
            out_d[q, t] = hi[srt[t]]; out_s[q, t] = hs[srt[t]]
        for t in range(size, K):
            out_d[q, t] = -1; out_s[q, t] = 0.0


@nb.njit(parallel=True, cache=True)
def _query_par(qptr, qids, idf, df, post_ptr, post_doc, doc_norm, max_df, min_keep, K, nblk, out_d, out_s, work):
    nq = len(qptr) - 1
    for b in nb.prange(nblk):
        q0 = b * nq // nblk; q1 = (b + 1) * nq // nblk
        _query_block(q0, q1, qptr, qids, idf, df, post_ptr, post_doc, doc_norm, max_df, min_keep, K, out_d, out_s, work)


class Index:
    def __init__(self, strings, mode):
        self.mode = mode
        buf, off = to_buf(strings)
        ip, ids = featurize(buf, off, mode)
        self.df, self.idf, self.pp, self.pd, self.dn = build_index(ip, ids, NFEAT[mode])
        self.n = len(off) - 1; self.nnz = len(ids)

    def query(self, strings, K=50, max_df=None, min_keep=0, nblk=None):
        buf, off = to_buf(strings)
        qp, qi = featurize(buf, off, self.mode)
        nq = len(off) - 1
        max_df = self.n if max_df is None else int(max_df)
        out_d = np.empty((nq, K), dtype=np.int32); out_s = np.empty((nq, K), dtype=np.float32)
        work = np.zeros(nq, dtype=np.int64)
        nblk = nblk or max(1, min(nb.get_num_threads(), nq))
        _query_par(qp, qi, self.idf, self.df, self.pp, self.pd, self.dn, max_df, int(min_keep), K, nblk, out_d, out_s, work)
        return out_d, out_s, work
