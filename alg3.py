import numpy as np
import alg2

np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=10000
)

def EpsilonEstimate(X, U, S, V, y, p, lambda_0):
    """
    Calculates the estimate of the noise amplitude 
        in the overparameterized and underparameterized case.

    Args:
        X: the design matrix with shape NxD.
        U: the matrix of left eigenvectors of X from its SVD decomposition.
        S: the singular value matrix of X from its SVD decomposition, 
            with singular values on the main diagonal.
        V: the matrix of right eigenvectors of X from its SVD decomposition.
        y: the truth vector with shape Nx1.
        p: the exponent parameter for the regularized rank.
        lambda_0: the initial ridge regression parameter, often set to 1.

    Returns:
        The estimate of the noise amplitude as a constant.
    """
    N, D = S.shape
    min_D_N = min(N, D)
    sigma = np.diag(S)[:min_D_N]
    r_p = np.sum(((sigma ** 2) / (sigma ** 2 + lambda_0)) ** p)

    if N > D:
        theta_hat = V[:, :min_D_N] @ ((sigma / (sigma **2 + lambda_0))[:, None] * (U[:, :min_D_N].T @ y))
        y_hat = X @ theta_hat
        epsilon_hat = np.sqrt((1/(N-1) * np.sum((y_hat - y) ** 2)) / (1 - r_p/N))
    else:
        h = N // 2
        X1, y1 = X[:h], y[:h]
        X2, y2 = X[h:], y[h:]

        # fit on fold 1, predict fold 2 (out-of-sample)
        U1, s1, V1 = np.linalg.svd(X1, full_matrices=False)
        S1 = np.diag(s1)
        N1, D1 = S1.shape
        theta1 = V1 @ np.linalg.pinv(S1.T @ S1 + lambda_0 * np.eye(N1, D1)) @ U1.T @ y1
        y_hat2 = X2 @ theta1

        # fit on fold 2, predict fold 1 (out-of-sample)
        U2, s2, V2 = np.linalg.svd(X2, full_matrices=False)
        S2 = np.diag(s2)
        N2, D2 = S2.shape
        theta2 = V2 @ np.linalg.pinv(S2.T @ S2 + lambda_0 * np.eye(N2, D2)) @ U2.T @ y2
        y_hat1 = X1 @ theta2

        mse_oos = (1/N) * (np.sum((y_hat2 - y2) ** 2) + np.sum((y_hat1 - y1) ** 2))
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

    theta_hat = V @ np.linalg.pinv(S.T @ S + lambda_0 * np.eye(D)) @ S.T @ U.T @ y

    epsilon_hat = EpsilonEstimate(X, U, S, V, y, p, lambda_0)

    return alg2.ModelOptReg(S, V, theta_hat, epsilon_hat, lambda_0, delta)
