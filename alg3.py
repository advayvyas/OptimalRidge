import numpy as np
import alg2

np.set_printoptions(
    precision=3,
    suppress=True,
    linewidth=10000
)

rng = np.random.default_rng(seed=77)

def EpsilonEstimate(S, theta_hat, p): # equation 14, 15
    fd

def SampleOptReg(X, y, lambda_0, p, delta):
    U, S, Vt = np.linalg.svd(X)
    V = Vt.T

    theta_hat = V @ np.linalg.pinv(S + (lambda_0 @ np.eye(N, D))) @ (U.T) @ y
    epsilon_hat = EpsilonEstimate(S, theta_hat, p)

    return ModelOptReg(S, V, theta_hat, epsilon_hat, lambda_0, delta)

if __name__ == "__main__":
    
