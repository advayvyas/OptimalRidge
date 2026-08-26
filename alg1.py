import numpy as np

np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=10000
)

def GenXData(N, D, sigma):
    """
    Generates a random mean-centered X from a multivariate normal distribution.

    Args:
        N: the number of observations
        D: the number of parameters
        sigma: the covariance matrix of the data (X)

    Returns:
        A matrix X of the shape NxD (N rows, D columns).
    """
    rng = np.random.default_rng(seed=77)

    mu = np.zeros(D)
    X = rng.multivariate_normal(mean = mu, cov = sigma, size = N)
    return X - X.mean(axis=0)

def GenYData(X, N, theta, epsilon):
    """
    Generates a random mean-centered y from a standard normal distribution.

    Args:
        N: the number of observations
        D: the number of parameters
        theta: the true parameter vector
        epsilon: the noise amplitude

    Returns:
        A column vector y of the shape Nx1 (N rows, 1 column).
    """
    rng = np.random.default_rng(seed=77)

    z = (rng.standard_normal(size = N))[:, np.newaxis]
    y = X @ theta + epsilon * z
    return y - y.mean(axis=0)

def GenData(sigma, theta, N, D, epsilon):
    """
    Generates random mean-centered X and y from GenXData, GenYData.

    Args:
        sigma: the covariance matrix of the data (X)
        theta: the true parameter vector
        N: the number of observations
        D: the number of parameters
        epsilon: the noise amplitude

    Returns:
        A matrix X (NxD) and vector y (Nx1).
    """
    X = GenXData(N, D, sigma)
    y = GenYData(X, N, theta, epsilon)
    return X, y

if __name__ == "__main__":
    rng_trial = np.random.default_rng(seed=77)

    N_TRIAL = 6
    D_TRIAL = 8
    sigma_trial = np.eye(D_TRIAL)
    theta_trial = (rng_trial.standard_normal(size=D_TRIAL))[:, np.newaxis]
    EPSILON = 6.767

    X_trial, y_trial = GenData(N = N_TRIAL, D = D_TRIAL, sigma = sigma_trial,
        theta = theta_trial, epsilon = EPSILON)

    print(X_trial)
    print(y_trial)
