import numpy as np
import alg2

np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=10000
)

def EpsilonEstimate(S, y, y_hat, p, lambda_0):
    """
    Calculates the estimate of the noise amplitude 
        in the overparameterized and undeparameterized case.

    Args:
        S: the singular value matrix of X from its SVD decomposition, 
            with singular values on the main diagonal.
        y: the truth vector with shape Nx1.
        y_hat: the prediction vector with shape Nx1.
        p: the exponent parameter for the regularized rank.
        lambda_0: the initial ridge regression parameter, often set to 1.

    Returns:
        The estimate of the noise amplitude as a constant.
    """
    min_D_N = min(S.shape)
    sigma = np.diag(S)[:min_D_N]
    N, D = S.shape

    r_p = np.sum(((sigma ** 2) / (sigma ** 2 + lambda_0)) ** p)

    if N > D:
        epsilon_hat = np.sqrt((1/(N-1) * np.sum((y_hat - y) ** 2)) / (1 - r_p/N)) # equation 14
    else: # manual 2-fold CV by series indices
        h = N // 2
        mse_oos = (1 / N) * (np.sum((y_hat[h:N] - y[0:h]) ** 2)
            + np.sum((y_hat[0:h] - y[h:N]) ** 2))
        epsilon_hat = np.sqrt((N/(N-1) * mse_oos) / (1 - r_p/N))

    return epsilon_hat

def SampleOptReg(U, S, V, y, lambda_0, p, delta):
    """
    Calculates the ridge regression parameter through a 
        fixed-point method, with approximations from the sample used.

    Args:
        U: the matrix of left eigenvectors of X from its SVD decomposition.
        S: the singular value matrix of X from its SVD decomposition, 
            with singular values on the main diagonal.
        V: the matrix of right eigenvectors of X from its SVD decomposition.
        y: the truth vector with shape Nx1.
        lambda_0: the initial ridge regression parameter, often set to 1.
        p: the exponent parameter for the regularized rank.
        delta: the iteration step size from lambda to lambda, often set to 10^{-4}.

    Returns:
        The approximate optimal lambda and its corresponding MSE value as a tuple.
    """

    X = U @ S @ (V.T)
    N, D = S.shape

    theta_hat = V @ np.linalg.pinv(S + (lambda_0 * np.eye(N, D))) @ (U.T) @ y
    y_hat = X @ theta_hat

    epsilon_hat = EpsilonEstimate(S, y, y_hat, p, lambda_0 = 1)

    return alg2.ModelOptReg(S, V, theta_hat, epsilon_hat, lambda_0, delta)
