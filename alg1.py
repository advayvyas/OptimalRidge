import numpy as np

np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=10000
)

def algorithm2(N = 100, d = 150):
    
    theta = (rng.standard_normal(size = d))[:, np.newaxis]
    # print(theta)
    # print((X[2]).T @ theta)
    
    z = (rng.standard_normal(size = N))[:, np.newaxis]
    # print(z)

    epsilon = 6.767
    # print(epsilon * z)

    y = X @ theta + epsilon * z
    # print(y)

def GenXData(N=100, d=120, Sigma=None):
    if Sigma is None:
        Sigma = np.eye(d)
    elif Sigma.shape != (d, d):
        raise ValueError(f"Sigma must be {d}x{d}, got {Sigma.shape}")

    rng = np.random.default_rng(seed=77)

    mu = np.zeros(d)
    
    X = rng.multivariate_normal(mean = mu, cov = Sigma, size = N)
    # print(X)

    return ((np.eye(N) - 1/N * np.ones(N) @ np.ones(N).T) @ X)

N = 6
d = 8
Sigma = np.eye(d)
GenXData(N, d, Sigma)

