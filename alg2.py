import numpy as np

np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=10000
)

def H(S, V, theta, lambda_):
    """
    Calculates the H of the given lambda where H(lambda) is 
        related to lambda by a factor of amplitude squared.
    
    Args:
        S: the singular value matrix of X from its SVD decomposition, 
            with singular values on the main diagonal.
        V: the matrix of right eigenvectors of X from its SVD decomposition.
        theta: the true parameter vector.
        lambda_: the ridge regression parameter.
    
    Returns:
        The value of H(lambda) as a constant.
    """
    min_D_N = min(S.shape)
    sigma = np.diag(S)[:min_D_N]

    projections = (V[:, :min_D_N].T @ theta).ravel()

    top = np.sum((sigma ** 4) / (sigma ** 2 + lambda_) ** 3)
    bottom = np.sum((sigma ** 4) * (projections ** 2) / (sigma ** 2 + lambda_) ** 3)

    return top / bottom

def mse(S, V, theta, lambda_, epsilon):
    """
    Calculates the mean squared error of the given parameters, in particular lambda.

    Args:
        S: the singular value matrix of X from its SVD decomposition, 
            with singular values on the main diagonal.
        V: the matrix of right eigenvectors of X from its SVD decomposition.
        theta: the true parameter vector.
        lambda_: the ridge regression parameter.
        epsilon: the noise amplitude.

    Returns:
        The expected out-of-sample mean squared error as a constant.
    """
    min_D_N = min(S.shape)
    sigma = np.diag(S)[:min_D_N]

    projections = (V[:, :min_D_N].T @ theta).ravel()

    bias_squared = np.sum((lambda_ * sigma * projections / (sigma ** 2 + lambda_)) ** 2)
    variance = (epsilon ** 2) * np.sum(((sigma ** 2) / (sigma ** 2 + lambda_)) ** 2)

    return (1 / S.shape[0]) * (bias_squared + variance) + (epsilon ** 2)

def ModelOptReg(S, V, theta, epsilon, lambda_0, delta):
    """
    Calculates the optimal ridge regression parameter through a fixed-point method.

    Args:
        S: the singular value matrix of X from its SVD decomposition, 
            with singular values on the main diagonal.
        V: the matrix of right eigenvectors of X from its SVD decomposition.
        theta: the true parameter vector.
        epsilon: the noise amplitude.
        lambda_0: the initial regression parameter, often set to 1.
        delta: the iteration step size from lambda to lambda, often set to 10^{-4}.

    Returns:
        The optimal lambda and its corresponding MSE value as a tuple.
    """
    lambda_ = lambda_0
    lambda_p = lambda_0 + 2 * delta

    while np.abs(lambda_ - lambda_p) > delta:
        lambda_p = lambda_
        lambda_ = (epsilon ** 2) * H(S, V, theta, lambda_)

    return lambda_, mse(S, V, theta, lambda_, epsilon)

if __name__ == "__main__":
    # trial
    rng = np.random.default_rng(seed=77)

    N_TRIAL = 1000
    D_TRIAL = 2000
    S_trial = np.eye(N_TRIAL, D_TRIAL)
    V_trial = np.eye(D_TRIAL)
    theta_trial = (rng.standard_normal(size=D_TRIAL))[:, np.newaxis]
    LAMBDA_0 = 1
    EPSILON = 6.767
    DELTA = 10 ** -4

    lambda_trial, mse_value = ModelOptReg(S_trial, V_trial,
        theta_trial, EPSILON, LAMBDA_0, DELTA)

    print(lambda_trial)
    print(mse_value)
