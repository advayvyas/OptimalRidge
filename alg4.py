import numpy as np
import alg1
import alg2
import alg3

np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=10000
)

def GlobalMinLambda(S, V, theta, epsilon):
    lambdas = np.linspace(-1, 10 ** 6, 1000)
    mse_values = np.array([alg2.mse(S, V, theta, lam, epsilon) for lam in lambdas])

    optimal_mse_value = np.min(mse_values)

    optimal_lambda_idx = np.argmin(mse_values)
    optimal_lambda = lambdas[optimal_lambda_idx]

    return optimal_lambda, optimal_mse_value

def EvaluateFixedX(N, D, sigma, epsilon, m_theta, m_X, m_y):
    lambda_0 = 1
    delta = 10 ** -4
    p = 1/4
    rng = np.random.default_rng(seed=77)

    lambda_min = np.zeros(shape = (m_X, m_theta))
    mse_min = np.zeros(shape = (m_X, m_theta))

    lambda_fixed_pt = np.zeros(shape = (m_X, m_theta))
    mse_fixed_pt = np.zeros(shape = (m_X, m_theta))

    lambda_sample_fixed_pt = np.zeros(shape = (m_X, m_theta, m_y))
    mse_sample_fixed_pt = np.zeros(shape = (m_X, m_theta, m_y))

    lambda_signal_noise = np.zeros(shape = (m_X, m_theta, m_y))
    mse_signal_noise = np.zeros(shape = (m_X, m_theta, m_y))

    for i_theta in range(m_theta):
        theta = rng.standard_normal(size=D)[:, np.newaxis]
        theta = theta / np.linalg.norm(theta)

        for i_X in range(m_X):
            X = alg1.GenXData(N, D, sigma)
            U, S, Vt = np.linalg.svd(X) # pylint: disable=unused-variable
            V = V.T

            lambda_min[i_X, i_theta], mse_min[i_X, i_theta] = GlobalMinLambda(S, V, theta, epsilon)
            lambda_fixed_pt[i_X, i_theta], mse_fixed_pt[i_X, i_theta] = alg2.ModelOptReg(S,
                V, theta, epsilon, lambda_0, delta)

            for i_y in range(m_y):
                y = alg1.GenYData(X, N, theta, epsilon)

                lambda_sample_fixed_pt[i_X, i_theta, i_y], mse_sample_fixed_pt[i_X,
                    i_theta, i_y] = alg3.SampleOptReg(X, y, lambda_0, p, delta)

                # TODO lambda signal to noise, mse from 4.2.3

    # TODO return BC-a bootstrapped 95% CI for lambda, mse values

if __name__ == "__main__":
    # trial
    N_TRIAL = 6
    D_TRIAL = 8
    sigma_trial = np.eye(D_TRIAL)
    M_THETA = 100
    M_X = 50
    M_Y = 50
    EPSILON = 6.767
