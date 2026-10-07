"""
Robust GMM-based threshold detection for imbalanced binary classification.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt


def _logsumexp(x):
    """Numerically stable log-sum-exp."""
    x_max = np.max(x, axis=-1, keepdims=True)
    return x_max.squeeze() + np.log(np.sum(np.exp(x - x_max), axis=-1))


def _fit_gmm(X, K=2, max_iter=100, tol=1e-6, reg=1e-6,
             entropy_alpha=1.0, n_init=1, verbose=False, print_func=print,
             rng=None):
    """Fit K-component GMM with entropy regularization.

    ``rng`` is a ``numpy.random.Generator`` used for the random restarts. It is
    intentionally decoupled from the global ``np.random`` state so the fitted
    threshold depends only on the score data, not on whatever seed the caller
    set for other stochastic parts of the pipeline (e.g. reviewer errors).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(X)
    best_ll = -np.inf
    best_params = None

    for run in range(n_init):
        # Initialize: use percentiles for first run, random for others
        if run == 0:
            percentiles = np.linspace(0, 100, K + 2)[1:-1]
            mu = np.percentile(X, percentiles)
        else:
            indices = rng.choice(n, size=K, replace=False)
            mu = np.sort(X[indices])

        sigma2 = np.full(K, np.var(X) / K)
        pi = np.full(K, 1.0 / K)

        # EM iterations
        ll_old = -np.inf
        for _ in range(max_iter):
            # E-step: compute responsibilities
            log_resp = np.array([stats.norm.logpdf(X, mu[k], np.sqrt(sigma2[k]))
                                 for k in range(K)]).T
            log_resp += np.log(pi)
            ll = np.sum(_logsumexp(log_resp))

            gamma = np.exp(log_resp - _logsumexp(log_resp)[:, None])

            # Check convergence
            if abs(ll - ll_old) < tol:
                break
            ll_old = ll

            # M-step: update parameters
            N_k = np.sum(gamma, axis=0)

            # Entropy regularization on mixing weights
            if entropy_alpha == 1.0:
                pi = N_k / n
            else:
                beta = 1.0 / entropy_alpha
                pi = N_k ** beta
                pi /= pi.sum()

            mu = np.sum(gamma * X[:, None], axis=0) / N_k
            sigma2 = np.sum(gamma * (X[:, None] - mu)**2, axis=0) / N_k + reg

        if ll > best_ll:
            best_ll = ll
            best_params = (pi.copy(), mu.copy(), sigma2.copy())

    return best_params


def _compute_f1(threshold, pi, mu, sigma2):
    """Compute predicted F1 score at threshold (K=2 only)."""
    cdf = [stats.norm.cdf(threshold, mu[k], np.sqrt(sigma2[k])) for k in range(2)]

    TP = pi[1] * (1 - cdf[1])
    FP = pi[0] * (1 - cdf[0])
    FN = pi[1] * cdf[1]

    if TP == 0:
        return 0.0

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return 2 * precision * recall / (precision + recall)


def _compute_pr(threshold, pi, mu, sigma2):
    """Predicted (precision, recall) at threshold from the fitted 2-component GMM.
    These are the *separate* ingredients of the F1 above; the review-confidence
    auto needs them apart (precision deficit -> over-merge -> gentler reviews;
    recall deficit -> under-connection -> more decisive reviews)."""
    cdf = [stats.norm.cdf(threshold, mu[k], np.sqrt(sigma2[k])) for k in range(2)]
    TP = pi[1] * (1 - cdf[1])
    FP = pi[0] * (1 - cdf[0])
    FN = pi[1] * cdf[1]
    if TP <= 0:
        return float('nan'), float('nan')
    return TP / (TP + FP), TP / (TP + FN)


def _find_threshold(pi, mu, sigma2):
    """Find threshold maximizing predicted F1 score."""
    mu_min, mu_max = mu.min(), mu.max()
    range_width = mu_max - mu_min

    result = minimize_scalar(
        lambda t: -_compute_f1(t, pi, mu, sigma2),
        bounds=(mu_min - 0.5 * range_width, mu_max + 0.5 * range_width),
        method='bounded'
    )

    return result.x, -result.fun


def _plot_threshold(scores, pi, mu, sigma2, threshold, plot_path):
    """Plot histogram with fitted distributions and threshold.

    Uses log scale on y-axis so the tiny positive component (often <1% of data)
    is visible alongside the dominant negative component.
    """
    fig, (ax_lin, ax_log) = plt.subplots(2, 1, figsize=(12, 10))

    # Generate x values for plotting distributions
    x_min, x_max = scores.min(), scores.max()
    x_range = x_max - x_min
    x = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 1000)

    # Unweighted PDFs so both components are visible regardless of mixing weight
    component_pdfs = []
    for k in range(len(pi)):
        pdf = stats.norm.pdf(x, mu[k], np.sqrt(sigma2[k]))
        component_pdfs.append(pdf)

    colors = ['red', 'green', 'blue', 'orange']
    labels = [f'Component {k+1} (μ={mu[k]:.3f}, σ={np.sqrt(sigma2[k]):.3f}, π={pi[k]:.4f})'
              for k in range(len(pi))]

    for ax, yscale, title_suffix in [
        (ax_lin, 'linear', '(linear scale)'),
        (ax_log, 'log', '(log scale — shows small component)')
    ]:
        ax.hist(scores, bins=200, density=True, alpha=0.5, color='gray',
                edgecolor='none', label='Data')

        for k in range(len(pi)):
            ax.plot(x, component_pdfs[k], color=colors[k % len(colors)],
                    linewidth=2, label=labels[k])

        ax.axvline(threshold, color='black', linestyle='--', linewidth=2,
                   label=f'Threshold = {threshold:.4f}')

        ax.set_yscale(yscale)
        ax.set_xlabel('Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'GMM Threshold Detection {title_suffix}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)

        if yscale == 'log':
            ax.set_ylim(bottom=1e-4)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_fallback_threshold(scores, threshold, plot_path, percentile):
    """Plot histogram with quantile-based fallback threshold (no GMM components)."""
    fig, (ax_lin, ax_log) = plt.subplots(2, 1, figsize=(12, 10))
    for ax, yscale, suffix in [(ax_lin, 'linear', '(linear scale)'),
                                (ax_log, 'log', '(log scale)')]:
        ax.hist(scores, bins=200, density=True, alpha=0.5, color='gray',
                edgecolor='none', label='Data')
        ax.axvline(threshold, color='black', linestyle='--', linewidth=2,
                   label=f'Fallback threshold = {threshold:.4f} (p{percentile})')
        ax.set_yscale(yscale)
        if yscale == 'log':
            ax.set_ylim(bottom=1e-4)
        ax.set_xlabel('Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'GMM Fit Failed — Quantile Fallback {suffix}',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()


def _fit_is_degenerate(pi, mu, sigma2):
    """Check whether the GMM parameters are invalid (NaN, non-positive variance, etc.)."""
    for arr in (pi, mu, sigma2):
        if not np.all(np.isfinite(arr)):
            return True
    if np.any(np.asarray(sigma2) <= 0):
        return True
    return False


def find_threshold(scores, entropy_alpha=0.85, n_init=1, verbose=False,
                   print_func=print, plot_path=None, fallback_percentile=99,
                   max_implied_K=None, random_state=0):
    """
    Find optimal classification threshold from unlabeled scores.

    Fits 2-component GMM with entropy regularization to handle extreme imbalance.
    Automatically detects and handles left-tail distributions.

    When the GMM fit degenerates (NaN weights, collapsed components, non-finite
    predicted F1), falls back to a percentile-based threshold.

    Parameters
    ----------
    scores : array-like
        Classifier scores (combined positive and negative samples)
    entropy_alpha : float, default=0.9
        Entropy regularization: < 1 favors sparse mixtures (bulk + tail)
    n_init : int, default=1
        Number of random initializations
    verbose : bool, default=False
        Print fitting progress
    plot_path : str, optional
        Path to save histogram plot with fitted distributions and threshold
    fallback_percentile : float, default=99
        Percentile used as the threshold when the GMM fit fails.
    max_implied_K : int, optional
        Reject any GMM fit whose threshold implies K = 1/pi_positive larger
        than this cap. When set, also replaces the fallback path with a
        percentile threshold that gives K = max_implied_K by construction.
        Use this when the embedding's positive/negative distributions overlap
        so heavily that the GMM hallucinates a tiny right-tail component.

    Returns
    -------
    threshold : float
        Optimal decision threshold
    """
    scores = np.asarray(scores).ravel()

    # Dedicated RNG for the GMM restarts — decoupled from global np.random so the
    # threshold is reproducible across runs that differ only in an unrelated seed
    # (e.g. reviewer-error seeds in the imperfect-reviewer experiment).
    rng = np.random.default_rng(random_state)

    def _use_fallback(reason):
        fallback = float(np.percentile(scores, fallback_percentile))
        print_func(f"⚠ {reason}; falling back to p{fallback_percentile} "
                   f"threshold = {fallback:.4f}")
        if plot_path is not None:
            _plot_fallback_threshold(scores, fallback, plot_path, fallback_percentile)
            print_func(f"📊 Saved fallback plot to: {plot_path}")
        find_threshold._last_predicted_f1 = float('nan')
        find_threshold._last_predicted_precision = float('nan')
        find_threshold._last_predicted_recall = float('nan')
        return fallback

    # Fit K=2 GMM
    k2_result = _fit_gmm(scores, K=2, entropy_alpha=entropy_alpha,
                         n_init=n_init, verbose=verbose, print_func=print_func,
                         rng=rng)
    if k2_result is None:
        return _use_fallback("K=2 GMM did not converge (all inits produced NaN log-likelihood)")
    pi, mu, sigma2 = k2_result

    # Safety check: if small component is LEFT of large component,
    # we have a left tail → refit with K=3 and use last 2 components
    if pi[0] < pi[1] and mu[0] < mu[1]:
        if verbose:
            print_func("⚠ Detected left tail, refitting with K=3...")

        k3_result = _fit_gmm(scores, K=3, entropy_alpha=0.1,
                             n_init=max(5, n_init), verbose=verbose, print_func=print_func,
                             rng=rng)
        if k3_result is None:
            # K=3 refit failed (numerically unstable with entropy_alpha=0.1);
            # keep the K=2 result and let the degeneracy guard / threshold
            # optimization handle whatever it produced.
            print_func("⚠ K=3 refit did not converge; keeping K=2 fit")
        else:
            pi3, mu3, sigma2_3 = k3_result
            # Use components 2 and 3 (ignore left tail)
            pi = pi3[1:3]
            pi_sum = pi.sum()
            if pi_sum > 0 and np.isfinite(pi_sum):
                pi = pi / pi_sum
            mu = mu3[1:3]
            sigma2 = sigma2_3[1:3]

        # Fallback: if bulk is still rightmost, fit to empirical right tail
        if pi[0] < pi[1] and mu[0] < mu[1]:
            bulk_pi = pi[1]
            bulk_mu = mu[1]
            bulk_sigma2 = sigma2[1]
            bulk_sigma = np.sqrt(bulk_sigma2)

            # Use empirical right-tail scores: only scores extremely unlikely under bulk
            # (99.9th percentile of bulk distribution = ~3.1 sigma above bulk mean)
            tail_cutoff = stats.norm.ppf(0.999, bulk_mu, bulk_sigma)
            right_tail = scores[scores > tail_cutoff]

            if len(right_tail) >= 10:
                if verbose:
                    print_func(f"⚠ Could not find right tail via GMM, fitting to {len(right_tail)} empirical right-tail scores...")
                fake_mu = np.mean(right_tail)
                fake_sigma2 = max(np.var(right_tail), bulk_sigma2 * 0.01)
                fake_pi = max(len(right_tail) / len(scores), 0.001)
            else:
                # Distribution is genuinely single-moded: place fake positive
                # component at the configured fallback_percentile of the data.
                if verbose:
                    print_func(f"⚠ Right tail essentially empty, using p{fallback_percentile} as threshold...")
                fake_mu = float(np.percentile(scores, fallback_percentile))
                fake_sigma2 = bulk_sigma2 * 0.01
                fake_pi = 1.0 - bulk_pi

            pi = np.array([bulk_pi, fake_pi])
            pi /= pi.sum()
            mu = np.array([bulk_mu, fake_mu])
            sigma2 = np.array([bulk_sigma2, fake_sigma2])

    # Guard: detect degenerate GMM fit (NaN weights, zero variance) before optimizing
    if _fit_is_degenerate(pi, mu, sigma2):
        return _use_fallback(
            f"GMM fit produced invalid parameters (pi={pi}, mu={mu}, sigma2={sigma2})"
        )

    # Find optimal threshold
    threshold, predicted_f1 = _find_threshold(pi, mu, sigma2)

    # Guard: F1 may be NaN even with finite params (e.g., fully overlapping components)
    if not np.isfinite(predicted_f1) or not np.isfinite(threshold):
        return _use_fallback("GMM predicted F1 or threshold is non-finite")

    # Guard: degenerate-but-finite fits where Component 2's mass collapses to ~0.
    # _compute_f1 returns 0.0 (not NaN) when TP = pi[1] * (1 - cdf[1]) is zero,
    # which happens when one component effectively has no mass. Threshold is then
    # whatever the bounded optimizer landed on across a flat F1=0 landscape — meaningless.
    if predicted_f1 < 1e-6:
        return _use_fallback(
            f"GMM predicted F1 collapsed to {predicted_f1:.2e} "
            f"(component mass degenerated; pi={pi})"
        )

    # Guard: GMM may "succeed" but produce a threshold so high that
    # 1/pi_positive exceeds any plausible cluster count — this happens when
    # the embedding's positive/negative distributions overlap heavily and the
    # K=3 / empirical-right-tail fallback latches onto a microscopic right tail.
    if max_implied_K is not None:
        pi_pos_emp = float(np.mean(scores > threshold))
        implied_K = (1.0 / pi_pos_emp) if pi_pos_emp > 0 else float('inf')
        if implied_K > max_implied_K:
            return _use_fallback(
                f"GMM threshold {threshold:.4f} implies K={implied_K:.0f} > "
                f"max_implied_K={max_implied_K} (pi_pos={pi_pos_emp:.6f})"
            )

    _pred_prec, _pred_rec = _compute_pr(threshold, pi, mu, sigma2)
    find_threshold._last_predicted_precision = _pred_prec
    find_threshold._last_predicted_recall = _pred_rec
    # Stash the fitted 2-component model (idx 0 = negative/incorrect, 1 = positive/correct)
    # so callers can form the likelihood ratio p(s|correct)/p(s|incorrect) and place a
    # human verdict on the classifier's evidence scale (Chuck's calibration idea).
    find_threshold._last_components = {'pi': list(map(float, pi)),
                                       'mu': list(map(float, mu)),
                                       'sigma2': list(map(float, sigma2))}
    print_func(f"✅ Optimal threshold: {threshold:.4f}, predicted F1: {predicted_f1:.4f} "
               f"(predicted precision={_pred_prec:.4f}, recall={_pred_rec:.4f})")

    # Generate plot if requested
    if plot_path is not None:
        _plot_threshold(scores, pi, mu, sigma2, threshold, plot_path)
        print_func(f"📊 Saved threshold plot to: {plot_path}")

    # Store predicted_f1 as a module-level variable for callers to access
    find_threshold._last_predicted_f1 = predicted_f1
    find_threshold._last_pi = pi
    find_threshold._last_mu = mu
    find_threshold._last_sigma2 = sigma2
    return threshold
