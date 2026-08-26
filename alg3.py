import numpy as np
import alg2

np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=10000
)

def EpsilonEstimate(X, y, y_hat, p, lambda_0): # pylint: disable=too-many-locals
    """
    Calculates the estimate of the noise amplitude 
        in the overparameterized and undeparameterized case.

    Args:
        X: the data matrix X with shape NxD.
        y: the truth vector with shape Nx1.
        y_hat: the prediction vector with shape Nx1.
        p: the exponent parameter for the regularized rank.
        lambda_0: the initial ridge regression parameter, often set to 1.

    Returns:
        The estimate of the noise amplitude as a constant.
    """
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
    """
    Calculates the ridge regression parameter through a 
        fixed-point method, with approximations from the sample used.

    Args:
        X: the data matrix X with shape NxD.
        y: the truth vector with shape Nx1.
        lambda_0: the initial ridge regression parameter, often set to 1.
        p: the exponent parameter for the regularized rank.
        delta: the iteration step size from lambda to lambda, often set to 10^{-4}.

    Returns:
        The approximate optimal lambda and its corresponding MSE value as a tuple.
    """

    U, S, Vt = np.linalg.svd(X)
    V = Vt.T

    N = S.shape[0]
    D = S.shape[1]

    theta_hat = V @ np.linalg.pinv(S + (lambda_0 @ np.eye(N, D))) @ (U.T) @ y
    y_hat = X @ theta_hat

    epsilon_hat = EpsilonEstimate(X, y, y_hat, p, lambda_0 = 1)

    return alg2.ModelOptReg(S, V, theta_hat, epsilon_hat, lambda_0, delta)
