import numpy as np
import alg1
import alg2
import alg3

np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=10000
)

def EvaluateFixedX(N, D, sigma, epsilon, m_theta, m_X, m_y):
    lambda_0 = 1
    delta = 10 ** -4
    p = 1/4
    
    for i_theta in range(m_theta):
        # uniform theta
        for i_X in range(m_X):
            X = alg1.GenXData(N, D, sigma)
            U, S, Vt = np.linalg.svd(X)
            V = V.T

            # min lambda from eq 16
            # mse from equation 8 and lambda min
            # lambda fixed point, mse using indexing with ModelOptReg

            for i_y in range(m_y):
                y = alg1.GenYData(X, theta, epsilon)
                # sample fixed point lambda from alg3.SampleOptReg
                # mse from equation 8 and lambda and indices
                # lambda signal to noise, mse from 4.2.3

    # return BC-a bootstrapped 95% CI for lambda, mse values
