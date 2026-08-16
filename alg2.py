import numpy as np

np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=10000
)

rng = np.random.default_rng(seed=77)

def H(S, V, theta, lambda_):
    min_d_N = min(S.shape)
    sigma = np.diag(S)[:min_d_N]

    projections = (V[:, :min_d_N].T @ theta).ravel()

    top = np.sum((sigma ** 4) / (sigma ** 2 + lambda_) ** 3)
    bottom = np.sum((sigma ** 4) * (projections ** 2) / (sigma ** 2 + lambda_) ** 3)

    return (top / bottom)

def mse(S, V, theta, lambda_, epsilon):
    min_d_N = min(S.shape)
    sigma = np.diag(S)[:min_d_N]

    projections = (V[:, :min_d_N].T @ theta).ravel()

    bias_squared = np.sum((lambda_ * sigma * projections / (sigma ** 2 + lambda_)) ** 2)
    variance = (epsilon ** 2) * np.sum(((sigma ** 2) / (sigma ** 2 + lambda_)) ** 2)

    return (1 / S.shape[0]) * (bias_squared + variance) + (epsilon ** 2)

def ModelOptReg(S, V, theta, epsilon, lambda_0, delta):
    lambda_ = lambda_0
    lambda_p = lambda_0 + 2 * delta

    while(np.abs(lambda_ - lambda_p) > delta):
        lambda_p = lambda_
        lambda_ = (epsilon ** 2) * H(S, V, theta, lambda_)

    return lambda_, mse(S, V, theta, lambda_, epsilon)

if __name__ == "__main__":
    # trial
    N = 1000
    d = 2000
    S = np.eye(N, d)
    V = np.eye(d)
    theta = (rng.standard_normal(size=d))[:, np.newaxis]
    lambda_0 = 1  
    epsilon = 6.767 
    delta = (10 ** -4)
    
    lambda_, mse_value = ModelOptReg(S, V, theta, epsilon, lambda_0, delta)
    print(lambda_)
    print(mse_value)

    # testing
    lambdas = np.logspace(-4, 2, 1000)

    mse_values = np.array([
        mse(S, V, theta, lam, epsilon)
        for lam in lambdas
    ])

    idx = np.argmin(mse_values)

    print("grid-search lambda:", lambdas[idx])
    print("grid-search MSE:", mse_values[idx])

    print("fixed-point lambda:", lambda_)
    print("fixed-point MSE:", mse_value)

    print("lambda =", lambda_)
    print("epsilon^2 H(lambda) =", epsilon**2 * H(S, V, theta, lambda_))
    print("difference =", abs(lambda_ - epsilon**2 * H(S, V, theta, lambda_)))