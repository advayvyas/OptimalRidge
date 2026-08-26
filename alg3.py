import numpy as np
import alg2

np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=10000
)

rng = np.random.default_rng(seed=77)

def EpsilonEstimate(X, y, y_hat, p, lambda_0):
    U, S, Vt = np.linalg.svd(X) # pylint: disable=unused-variable

    N = S.shape[0]
    D = S.shape[1]
    min_D_N = min(S.shape)
    sigma = np.diag(S)[:min_D_N]

    r_p = np.sum(((sigma ** 2) / (sigma ** 2 + lambda_0)) ** p)

    if N > D:
        epsilon_hat = np.sqrt((1/(N-1) * np.sum((y_hat - y) ** 2)) / (1 - r_p/N)) # equation 14
    else: # manual 2-fold CV by series indices
        h = N // 2
        mse_oos = 1/N [np.sum((y_hat[h:N] - y[0:h]) ** 2) + np.sum((y_hat[0:h] - y[h:N]) ** 2)]
        epsilon_hat = np.sqrt((N/(N-1) * mse_oos) / (1 - r_p/N))

    return epsilon_hat

def SampleOptReg(X, y, lambda_0, p, delta):
    U, S, Vt = np.linalg.svd(X)
    V = Vt.T

    N = S.shape[0]
    D = S.shape[1]

    theta_hat = V @ np.linalg.pinv(S + (lambda_0 @ np.eye(N, D))) @ (U.T) @ y
    y_hat = X @ theta_hat

    epsilon_hat = EpsilonEstimate(X, y, y_hat, p, lambda_0 = 1)

    return alg2.ModelOptReg(S, V, theta_hat, epsilon_hat, lambda_0, delta)

if __name__ == "__main__":
    print("code run")
