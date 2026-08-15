import numpy as np

np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=10000
)

def algorithm2(N = 100, d = 150):
    rng = np.random.default_rng(seed=77)

    mu = np.zeros(d)
    Sigma = np.eye(d)
    X = rng.multivariate_normal(mean = mu, cov = Sigma, size = N)
    # print(X)

    theta = (rng.standard_normal(size = d))[:, np.newaxis]
    # print(theta)
    # print((X[2]).T @ theta)
    
    z = (rng.standard_normal(size = N))[:, np.newaxis]
    # print(z)

    epsilon = 6.767
    # print(epsilon * z)

    y = X @ theta + epsilon * z
    # print(y)

algorithm2(N = 4, d = 5)