import numpy as np
from scipy.stats import norm

np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=10000
)

def bulk(D):
    """
    Bulk-type covariance eigenvalues

    Args:
        D: the number of parameters in the data matrix

    Returns:
        Sigma: the covariance matrix
    """

    j = np.arange(1, D + 1)
    eigenvalues = (D - j + 1) / D
    Sigma = np.diag(eigenvalues)

    return Sigma

def spiked(D):
    """
    Spiked-type covariance eigenvalues.

    Args:
        D: the number of parameters in the data matrix

    Returns:
        Sigma: the covariance matrix
    """

    x = np.linspace(-3, 3, D)
    eigenvalues = norm.pdf(x)
    eigenvalues = eigenvalues / eigenvalues.max()
    Sigma = np.diag(eigenvalues)

    return Sigma

if __name__ == "__main__":
    D_TRIAL = 10
    Sigma_bulk = bulk(D_TRIAL)
    Sigma_spiked = spiked(D_TRIAL)

    print("Bulk eigenvalues: \n", Sigma_bulk)
    print("Spiked eigenvalues: \n", Sigma_spiked)
