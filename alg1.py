import numpy as np

np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=10000
)

def GenXData(N, D, sigma):
    rng = np.random.default_rng(seed=77)

    mu = np.zeros(D)
    X = rng.multivariate_normal(mean = mu, cov = sigma, size = N)
    return X - X.mean(axis=0)

def GenYData(X, N, theta, epsilon):
    rng = np.random.default_rng(seed=77)

    z = (rng.standard_normal(size = N))[:, np.newaxis]
    y = X @ theta + epsilon * z
    return y - y.mean(axis=0)

def GenData(sigma, theta, N, D, epsilon):
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
