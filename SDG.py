# Stochastic Gradient Descent

from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np

#====================== Data

from DataRegression import X, X_test, X_train, x, x_mesh, y_mesh, z_test, z_train, plotFunction, z



def compute_square_loss(X, y, theta):
    loss = 0  # Initialize the average square loss

    m = len(y)
    loss = (1.0 / m) * (np.linalg.norm((X.dot(theta) - y)) ** 2)
    return loss


def gradient_ridge(X, y, beta, lambda_):
    return 2 * (np.dot(X.T, (X.dot(beta) - y))) + 2 * lambda_ * beta


def gradient_ols(X, y, beta):
    m = X.shape[0]

    grad = 2 / m * X.T.dot(X.dot(beta) - y)

    return grad


def learning_schedule(t):
    t0, t1 = 5, 50
    return t0 / (t + t1)


def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.random.permutation(inputs.shape[0])
    for start_idx in range(0, inputs.shape[0], batchsize):
        end_idx = min(start_idx + batchsize, inputs.shape[0])
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        yield inputs[excerpt], targets[excerpt]


###sgd
def SGD(X, y, learning_rate=0.02, n_epochs=100, lambda_=0.01, batch_size=5, method='ols'):
    num_instances, num_features = X.shape[0], X.shape[1]
    beta = np.random.randn(num_features)  ##initialize beta

    for epoch in range(n_epochs + 1):

        for batch in iterate_minibatches(X, y, batch_size, shuffle=True):

            X_batch, y_batch = batch

            # for i in range(batch_size):
            #     learning_rate = learning_schedule(n_epochs*epoch + i)

            if method == 'ols':
                gradient = gradient_ols(X_batch, y_batch, beta)
                beta = beta - learning_rate * gradient
            if method == 'ridge':
                gradient = gradient_ridge(X_batch, y_batch, beta, lambda_=lambda_)
                beta = beta - learning_rate * gradient

    mse_ols_train = compute_square_loss(X, y, beta)
    mse_ridge_train = compute_square_loss(X, y, beta) + lambda_ * np.dot(beta.T, beta)

    return beta


def compute_test_mse(X_test, y_test, beta, lambda_=0.01):
    mse_ols_test = compute_square_loss(X_test, y_test, beta)
    mse_ridge_test = compute_square_loss(X_test, y_test, beta) + lambda_ * np.dot(beta.T, beta)
    return mse_ols_test, mse_ridge_test


def plot_MSE_ols_ridge_nn():
    MSE_ridge_val = []
    MSE_ols_val = []

    methods = ['ridge', 'ols']
    for method in methods:

        if method == 'ridge':

            eta = np.logspace(-5, -3, 10)
            best_beta_ridge = eta[0]
            best_mse_ridge = 1e10

            for i in eta:
                beta = SGD(X_train, z_train, learning_rate=i, lambda_=0.01, method='ridge')
                mse_ols_, mse_ridge_ = compute_test_mse(X_test, z_test, lambda_=0.01, beta=beta)
                MSE_ridge_val.append(mse_ridge_)
                if mse_ridge_ < best_mse_ridge:
                    best_beta_ridge = beta
                    best_mse_ridge = mse_ridge_

        if method == 'ols':

            eta = np.logspace(-5, -3, 10)
            best_beta_ols = eta[0]
            best_mse_ols = 1e10

            for i in eta:
                beta = SGD(X_train, z_train, learning_rate=i, lambda_=0.01, method='ols')
                mse_ols_, mse_ridge_ = compute_test_mse(X_test, z_test, beta=beta)
                MSE_ols_val.append(mse_ols_)
                if mse_ols_ < best_mse_ols:
                    best_beta_ols = beta
                    best_mse_ols = mse_ols_
    print(MSE_ridge_val)
    print(MSE_ols_val)
    plot, ax = plt.subplots()
    plt.title('MSE for the OLS, Ridge and Neural Networks (validation data)')
    plt.semilogx(np.logspace(-5, -3, 10), MSE_ridge_val, 'k-o', label='Ridge')
    plt.semilogx(np.logspace(-5, -3, 10), MSE_ols_val, 'r-o', label='OLS')
    plt.xlabel(r'Learning rate $\eta$')
    plt.ylabel('MSE')
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend()
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9)
    print(best_beta_ols.shape)
    print(X.shape)
    best_pred_ols = X @ best_beta_ols
    best_pred_ridge = X @ best_beta_ridge
    plotFunction(x_mesh, y_mesh, z, 'data')
    plotFunction(x_mesh, y_mesh, best_pred_ols.reshape(len(x), len(x)), 'OLS')
    plotFunction(x_mesh, y_mesh, best_pred_ridge.reshape(len(x), len(x)), 'RIDGE')
    # plt.savefig(os.path.join(os.path.dirname(__file__), 'Plots', 'mse_all.png'), transparent=True, bbox_inches='tight')
    return plt.show()


plot_MSE_ols_ridge_nn()