import numpy as np

def fdrcorrection(pvals, Q=0.05):
    """Correct pvals using Benjamini-Hochberg procedure to control false discovery rate."""
    m = len(pvals)
    sorted = np.sort(pvals)
    ranks = np.arange(1, len(sorted)+1)
    crit = sorted < ranks / m * Q
    if not crit.any():
        # if no p-values fulfill the criterion, return only zeros (none are significant)
        return np.zeros(m, dtype=bool)
    max_p = np.max(sorted[crit])
    return pvals <= max_p
