import numpy as np

np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=10000
)

rng = np.random.default_rng(seed=77)

def GenXData(N=100, d=120, Sigma=None):
    if Sigma is None:
        Sigma = np.eye(d)
    elif Sigma.shape != (d, d):
        raise ValueError(f"Sigma must be {d}x{d}, got {Sigma.shape}")

    rng = np.random.default_rng(seed=77)

    mu = np.zeros(d)
    
    X = rng.multivariate_normal(mean = mu, cov = Sigma, size = N)

    return (X - X.mean(axis=0))

def GenYData(X, N, theta, epsilon):
    rng = np.random.default_rng(seed=77)

    z = (rng.standard_normal(size = N))[:, np.newaxis]

    y = X @ theta + epsilon * z

    return (y - y.mean(axis=0))
    
def algorithm(Sigma, theta, N = 10, d = 12, epsilon = 6.767):
    X = GenXData(N, d, Sigma)
    y = GenYData(X, N, theta, epsilon)
    return X, y

if __name__ == "__main__":
    N = 6
    d = 8
    Sigma = np.eye(d)
    theta = (rng.standard_normal(size=d))[:, np.newaxis]
    epsilon = 6.767

    X, y = algorithm(N = N, d = d, Sigma = Sigma, theta = theta, epsilon = epsilon)

    print(X)
    print(y)

