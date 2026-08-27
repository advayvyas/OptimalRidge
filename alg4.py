import numpy as np
import scipy
from scipy.optimize import minimize_scalar
from tqdm.auto import tqdm
import alg1
import alg2
import alg3

np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=10000
)

def GlobalMinLambda(S, V, theta, epsilon):
    """
    Finds the globally optimal ridge regression parameter 
        that minimizes the mean squared error (MSE).

    Args:
        S: the singular value matrix of X from its SVD decomposition, 
            with singular values on the main diagonal.
        V: the matrix of right eigenvectors of X from its SVD decomposition.
        theta: the true parameter vector.
        epsilon: the noise amplitude.

    Returns:
        The optimal parameter and its corresponding MSE value as a tuple.

    """
    optimal = minimize_scalar(lambda lam: alg2.mse(S, V, theta, lam, epsilon),
        bounds=(-1, 10**6), method='bounded', options={'xatol': 1e-8})
    optimal_lambda = optimal.x
    optimal_mse_value = optimal.fun

    return optimal_lambda, optimal_mse_value

def SignalNoiseLambda(U, S, V, y, lambda_0, p): # pylint: disable=too-many-locals
    """
    Finds the ridge regression parameter estimate 
        through a sample-based version of the signal-to-noise approach.
    
    Args:
        U: the matrix of left eigenvectors of X from its SVD decomposition.
        S: the singular value matrix of X from its SVD decomposition, 
            with singular values on the main diagonal.
        V: the matrix of right eigenvectors of X from its SVD decomposition.
        y: the truth vector with shape Nx1.
        lambda_0: the initial ridge regression parameter, often set to 1.
        p: the exponent parameter for the regularized rank.

    Returns:
        The estimated parameter and its corresponding MSE value as a tuple.
    """
    X = U @ S @ (V.T)
    N, D = S.shape

    theta_hat = V @ np.linalg.pinv(S + (lambda_0 * np.eye(N, D))) @ (U.T) @ y
    y_hat = X @ theta_hat

    epsilon_hat = alg3.EpsilonEstimate(S, y, y_hat, p, lambda_0 = 1)
    lambda_signal_noise = (D * (epsilon_hat ** 2)) / (np.sum(np.square(theta_hat)))

    return lambda_signal_noise, alg2.mse(S, V, theta_hat, lambda_signal_noise, epsilon_hat)

def EvaluateFixedX(N, D, Sigma, epsilon, m_theta, m_X, m_y): # pylint: disable=too-many-locals
    """
    Evaluates the global minimum, fixed-point, sample fixed-point, signal to noise, and 
        default methods of finding the regularization parameter and 
        tabulates their corresponding mean squared error (MSE) values.

    Args:
        N: the number of observations.
        D: the number of parameters.
        Sigma: the covariance matrix of the data matrix 
            X with shape DxD.
        epsilon: the noise amplitude.
        m_theta: the number of randomly sampled parameters.
        m_X: the number of random X samples.
        m_y: the number of random y samples from each pair of theta and X.

    Returns:
        A tuple containing bootstrap results for the median lambda and MSE
            from the global minimum, fixed-point, sample fixed-point, and
            signal-to-noise methods, followed by the bootstrap result for the
            default method's MSE. Each result is a scipy.stats.BootstrapResult.
    """
    lambda_0 = 1
    delta = 10 ** -4
    p = 1/4
    bootstrap_batch = 1024
    rng = np.random.default_rng(seed=77)

    lambda_default = 1
    mse_default = np.zeros(shape = (m_X, m_theta))

    lambda_min = np.zeros(shape = (m_X, m_theta))
    mse_min = np.zeros(shape = (m_X, m_theta))

    lambda_fixed_pt = np.zeros(shape = (m_X, m_theta))
    mse_fixed_pt = np.zeros(shape = (m_X, m_theta))

    lambda_sample_fixed_pt = np.zeros(shape = (m_X, m_theta, m_y))
    mse_sample_fixed_pt = np.zeros(shape = (m_X, m_theta, m_y))

    lambda_signal_noise = np.zeros(shape = (m_X, m_theta, m_y))
    mse_signal_noise = np.zeros(shape = (m_X, m_theta, m_y))

    # for i_theta in range(m_theta):
    for i_theta in tqdm(range(m_theta), desc="Evaluating theta values"): # pylint: disable=undefined-variable
        theta = rng.standard_normal(size=D)[:, np.newaxis]
        theta = theta / np.linalg.norm(theta)

        for i_X in range(m_X):
            X = alg1.GenXData(N, D, Sigma)
            U, s, Vt = np.linalg.svd(X) # pylint: disable=unused-variable
            V = Vt.T

            min_D_N = min(N, D)
            S = np.zeros((N, D))
            S[:min_D_N, :min_D_N] = np.diag(s)

            mse_default[i_X, i_theta] = alg2.mse(S, V, theta, lambda_default, epsilon)

            lambda_min[i_X, i_theta], mse_min[i_X,
                i_theta] = GlobalMinLambda(S, V, theta, epsilon)
            lambda_fixed_pt[i_X, i_theta], mse_fixed_pt[i_X, i_theta] = alg2.ModelOptReg(S,
                V, theta, epsilon, lambda_0, delta)

            for i_y in range(m_y):
                y = alg1.GenYData(X, N, theta, epsilon)

                lambda_sample_fixed_pt[i_X, i_theta, i_y], mse_sample_fixed_pt[i_X,
                    i_theta, i_y] = alg3.SampleOptReg(U, S, V, y, lambda_0, p, delta)

                lambda_signal_noise[i_X, i_theta, i_y], mse_signal_noise[i_X,
                    i_theta, i_y] = SignalNoiseLambda(U, S, V, y, lambda_0, p)

    lambda_min_bootstrap = scipy.stats.bootstrap((lambda_min.flatten(),),
        np.median, confidence_level = 0.95, method = 'BCa', batch = bootstrap_batch, rng = rng)
    mse_min_bootstrap = scipy.stats.bootstrap((mse_min.flatten(),),
        np.median, confidence_level = 0.95, method = 'BCa', batch = bootstrap_batch, rng = rng)

    lambda_fixed_pt_bootstrap = scipy.stats.bootstrap((lambda_fixed_pt.flatten(),),
        np.median, confidence_level = 0.95, method = 'BCa', batch = bootstrap_batch, rng = rng)
    mse_fixed_pt_bootstrap = scipy.stats.bootstrap((mse_fixed_pt.flatten(),),
        np.median, confidence_level = 0.95, method = 'BCa', batch = bootstrap_batch, rng = rng)

    lambda_sample_fixed_pt_bootstrap = scipy.stats.bootstrap((lambda_sample_fixed_pt.flatten(),),
        np.median, confidence_level = 0.95, method = 'BCa', batch = bootstrap_batch, rng = rng)
    mse_sample_fixed_pt_bootstrap = scipy.stats.bootstrap((mse_sample_fixed_pt.flatten(),),
        np.median, confidence_level = 0.95, method = 'BCa', batch = bootstrap_batch, rng = rng)

    lambda_signal_noise_bootstrap = scipy.stats.bootstrap((lambda_signal_noise.flatten(),),
        np.median, confidence_level = 0.95, method = 'BCa', batch = bootstrap_batch, rng = rng)
    mse_signal_noise_bootstrap = scipy.stats.bootstrap((mse_signal_noise.flatten(),),
        np.median, confidence_level = 0.95, method = 'BCa', batch = bootstrap_batch, rng = rng)

    mse_default_bootstrap = scipy.stats.bootstrap((mse_default.flatten(),),
        np.median, confidence_level = 0.95, method = 'BCa', batch = bootstrap_batch, rng = rng)

    return (lambda_min_bootstrap, mse_min_bootstrap,
                lambda_fixed_pt_bootstrap, mse_fixed_pt_bootstrap,
                lambda_sample_fixed_pt_bootstrap, mse_sample_fixed_pt_bootstrap,
                lambda_signal_noise_bootstrap, mse_signal_noise_bootstrap,
                mse_default_bootstrap)

if __name__ == "__main__":
    # trial
    N_TRIAL = 100
    ASPECT_RATIO = 0.9
    D_TRIAL = int(ASPECT_RATIO * N_TRIAL)
    Sigma_trial = np.eye(D_TRIAL)
    M_THETA = 100
    M_X = 50
    M_Y = 50
    EPSILON = 6.767

    results = EvaluateFixedX(N_TRIAL, D_TRIAL, Sigma_trial, EPSILON, M_THETA, M_X, M_Y)

    for index, result in enumerate(results, start=1):
        print(f"Result {index}:")
        print("Confidence interval:", result.confidence_interval)
        print("Standard error:", result.standard_error)
        print("Bootstrap distribution:")
        print(result.bootstrap_distribution)
