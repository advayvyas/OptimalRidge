import numpy as np

np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=10000
)

rng = np.random.default_rng(seed=77)

def GenXData(N=100, D=120, sigma=None):
    if sigma is None:
        sigma = np.eye(D)
    elif sigma.shape != (D, D):
        raise ValueError(f"Sigma must be {D}x{D}, got {sigma.shape}")

    rng = np.random.default_rng(seed=77)

    mu = np.zeros(D)
    
    X = rng.multivariate_normal(mean = mu, cov = sigma, size = N)

    return (X - X.mean(axis=0))

def GenYData(X, N, theta, epsilon):
    rng = np.random.default_rng(seed=77)

    z = (rng.standard_normal(size = N))[:, np.newaxis]

    y = X @ theta + epsilon * z

    return (y - y.mean(axis=0))
    
def algorithm(sigma, theta, N = 10, D = 12, epsilon = 6.767):
    X = GenXData(N, D, sigma)
    y = GenYData(X, N, theta, epsilon)
    return X, y

if __name__ == "__main__":
    N = 6
    D = 8
    sigma = np.eye(D)
    theta = (rng.standard_normal(size=D))[:, np.newaxis]
    epsilon = 6.767

    X, y = algorithm(N = N, D = D, sigma = sigma, theta = theta, epsilon = epsilon)

    print(X)
    print(y)

