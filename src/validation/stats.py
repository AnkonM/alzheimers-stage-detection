"""Statistics for the OASIS protocol (pure functions on arrays; no I/O, no labels read from disk).

AUC and its bootstrap, Jonckheere-Terpstra (one-sided, tie-corrected normal approximation), Spearman with a
bootstrap CI, partial Spearman, AUC stratified by bins, and the H1 verdict rule from the protocol.
"""
import numpy as np
from scipy import stats as sps

# ---- constants fixed in docs/external_validation_protocol.md section 7; defined ONCE, here. A test parses the
# ---- protocol text and asserts that every one of them matches.
SEED = 20261113                 # bootstrap seed
N_BOOT = 2000                   # bootstrap resamples
H1_PASS_AUC, H1_PASS_CI_LOW, H1_FAIL_AUC = 0.75, 0.65, 0.65
S2_ALPHA = 0.05                 # one-sided Jonckheere-Terpstra
S3_RHO_MAX = -0.30              # S3 passes when rho (ALZ vs MMSE) is at most this
AGE_RESTRICT = (60.0, 90.3)     # S6e: the ADNI training age range
S4_AGE_EDGES = (71, 80)         # S4 age-tertile edges fixed at Stage A: age <= 71, 71 < age <= 80, age > 80
S6F_N_CDR0 = 30                 # S6f: first 30 CDR 0 cases by sorted case id (plus every CDR >= 1 case)
CV_FOLDS, CV_REPEATS, LOGIT_L2 = 5, 20, 1.0     # S4 benchmarks: repeated stratified cross-validation


def auc(pos, neg):
    """Mann-Whitney AUC, P(score_pos > score_neg) with ties counted half. Higher score = positive class."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("both classes need at least one subject")
    if not (np.isfinite(pos).all() and np.isfinite(neg).all()):
        raise ValueError("scores must be finite (missing scores are handled by the QC policy, never here)")
    r = sps.rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bootstrap_auc(pos, neg, seed=SEED, n_boot=N_BOOT):
    """Subject-level percentile bootstrap, resampling positives and negatives separately (class sizes fixed).
    Vectorised: the pairwise comparison matrix is built once and each resample is a pair of multinomial weight
    vectors; the result depends only on the seed."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    point = auc(pos, neg)                                           # also validates the input
    n1, n0 = len(pos), len(neg)
    m = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    rng = np.random.default_rng(seed)
    ip, in_ = rng.integers(0, n1, (n_boot, n1)), rng.integers(0, n0, (n_boot, n0))
    wp = np.apply_along_axis(np.bincount, 1, ip, minlength=n1).astype(float)
    wn = np.apply_along_axis(np.bincount, 1, in_, minlength=n0).astype(float)
    a = np.einsum("bi,ij,bj->b", wp, m, wn) / (n1 * n0)
    lo, hi = np.percentile(a, [2.5, 97.5])
    return {"auc": point, "ci95": [float(lo), float(hi)], "n_pos": int(n1), "n_neg": int(n0),
            "seed": seed, "n_boot": n_boot}


def h1_verdict(auc_value, ci_low):
    """Protocol section 7: PASS / PARTIAL / FAIL (AUC below 0.5 is FAIL and flagged inverted)."""
    if auc_value >= H1_PASS_AUC and ci_low > H1_PASS_CI_LOW:
        return "PASS"
    if auc_value < H1_FAIL_AUC:
        return "FAIL"
    return "PARTIAL"


def jonckheere_terpstra(groups, alternative="increasing"):
    """groups: list of 1-D arrays ordered from lowest to highest expected level. Returns {J, z, p}; p is
    one-sided for the stated alternative. Normal approximation with the standard tie correction."""
    groups = [np.asarray(g, float) for g in groups]
    ns = np.array([len(g) for g in groups])
    if len(groups) < 2 or (ns < 1).any():
        raise ValueError("need at least two non-empty groups")
    J = 0.0
    for i in range(len(groups) - 1):
        for j in range(i + 1, len(groups)):
            x, y = groups[i][:, None], groups[j][None, :]
            J += float((y > x).sum() + 0.5 * (y == x).sum())
    N = ns.sum()
    mean = (N ** 2 - (ns ** 2).sum()) / 4.0
    allv = np.concatenate(groups)
    _, tcount = np.unique(allv, return_counts=True)
    t = tcount.astype(float)
    var = (N * (N - 1) * (2 * N + 5) - (ns * (ns - 1) * (2 * ns + 5)).sum() - (t * (t - 1) * (2 * t + 5)).sum()) / 72.0
    var += ((ns * (ns - 1) * (ns - 2)).sum() * (t * (t - 1) * (t - 2)).sum()) / (36.0 * N * (N - 1) * (N - 2)) if N > 2 else 0
    var += ((ns * (ns - 1)).sum() * (t * (t - 1)).sum()) / (8.0 * N * (N - 1))
    z = (J - mean) / np.sqrt(var)
    p = float(sps.norm.sf(z)) if alternative == "increasing" else float(sps.norm.cdf(z))
    return {"J": J, "z": float(z), "p": p, "alternative": alternative}


def spearman(x, y):
    r, p = sps.spearmanr(x, y)
    return float(r), float(p)


def bootstrap_spearman(x, y, seed=SEED, n_boot=N_BOOT):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    rs = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        rs[b] = sps.spearmanr(x[i], y[i])[0]
    lo, hi = np.nanpercentile(rs, [2.5, 97.5])
    r, p = spearman(x, y)
    return {"rho": r, "p": p, "ci95": [float(lo), float(hi)], "n": int(len(x)), "seed": seed, "n_boot": n_boot}


def partial_spearman(x, y, z):
    """Spearman correlation of x and y controlling for z (rank-transform, residualise on z, correlate)."""
    rx, ry, rz = (sps.rankdata(np.asarray(v, float)) for v in (x, y, z))
    A = np.column_stack([np.ones_like(rz), rz])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def stratified_auc(score, positive, strata, edges):
    """AUC within bins of `strata`. Bins are RIGHT-closed, as the protocol states for the age tertiles: with edges
    (e1, e2) the groups are strata <= e1, e1 < strata <= e2 and strata > e2. Bins lacking either class give None."""
    score, positive, strata = np.asarray(score, float), np.asarray(positive, bool), np.asarray(strata, float)
    bins = np.digitize(strata, edges, right=True)
    out = []
    for b in range(len(edges) + 1):
        m = bins == b
        pos, neg = score[m & positive], score[m & ~positive]
        out.append({"bin": b, "n_pos": int(len(pos)), "n_neg": int(len(neg)),
                    "auc": auc(pos, neg) if len(pos) and len(neg) else None})
    return out


def bootstrap_partial_spearman(x, y, z, seed=SEED, n_boot=N_BOOT):
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    rng = np.random.default_rng(seed)
    r = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        r[b] = partial_spearman(x[i], y[i], z[i])
    lo, hi = np.nanpercentile(r, [2.5, 97.5])
    return {"rho": partial_spearman(x, y, z), "ci95": [float(lo), float(hi)], "n": int(len(x)), "seed": seed, "n_boot": n_boot}


def fit_logistic(X, y, l2=LOGIT_L2, iters=100, tol=1e-10):
    """L2-penalised logistic regression by Newton's method (numpy only). Columns are standardised with the
    training data; the intercept is not penalised. Same objective as sklearn's LogisticRegression with C = 1 / l2.
    Returns a function mapping new rows to P(y = 1)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    A = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    pen = np.diag([0.0] + [float(l2)] * (A.shape[1] - 1))
    w = np.zeros(A.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(A @ w, -35, 35)))
        grad = A.T @ (p - y) + pen @ w
        H = (A * (p * (1 - p))[:, None]).T @ A + pen + 1e-12 * np.eye(len(w))
        step = np.linalg.solve(H, grad)
        w -= step
        if np.abs(step).max() < tol:
            break

    def predict(Xn):
        An = np.column_stack([np.ones(len(Xn)), (np.asarray(Xn, float) - mu) / sd])
        return 1.0 / (1.0 + np.exp(-np.clip(An @ w, -35, 35)))
    predict.coef = w
    return predict


def repeated_cv_auc(X, y, folds=CV_FOLDS, repeats=CV_REPEATS, seed=SEED, l2=LOGIT_L2, n_boot=N_BOOT):
    """Out-of-fold AUC of a logistic model from repeated stratified k-fold cross-validation, so the benchmark is
    not optimistic. Each repeat assigns folds within each class at random; every subject's prediction always comes
    from a model that did not see that subject. Reported: AUC of the repeat-averaged out-of-fold predictions with
    a subject-level bootstrap CI, and the mean and SD of the per-repeat AUCs."""
    X, y = np.asarray(X, float), np.asarray(y, int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    k = min(folds, n1, n0)
    if k < 2:
        raise ValueError("too few cases in one class for cross-validation")
    rng = np.random.default_rng(seed)
    oof = np.zeros((repeats, len(y)))
    for r in range(repeats):
        fold = np.empty(len(y), int)
        for cls in (0, 1):
            idx = rng.permutation(np.flatnonzero(y == cls))
            fold[idx] = np.arange(len(idx)) % k
        for f in range(k):
            te = fold == f
            model = fit_logistic(X[~te], y[~te], l2)
            oof[r, te] = model(X[te])
    per_repeat = [auc(oof[r][y == 1], oof[r][y == 0]) for r in range(repeats)]
    mean_oof = oof.mean(0)
    b = bootstrap_auc(mean_oof[y == 1], mean_oof[y == 0], seed=seed, n_boot=n_boot)
    return {"auc": b["auc"], "ci95": b["ci95"], "auc_per_repeat_mean": float(np.mean(per_repeat)),
            "auc_per_repeat_sd": float(np.std(per_repeat, ddof=1)), "folds": int(k), "repeats": int(repeats),
            "n_pos": n1, "n_neg": n0, "seed": seed, "l2": l2}
