from SDG import SDG
import numpy as np
import matplotlib.pyplot as plt



#====================== Data

from DataRegression import X, X_test, X_train, x, x_mesh, y_mesh, z_test, z_train, plotFunction, z





def plot_MSE_ols_ridge():
    MSE_ridge_val = []
    MSE_ols_val = []

    methods = ['ridge', 'ols']
    for method in methods:

        if method == 'ridge':

            eta = np.logspace(-5, -3, 10)
            best_beta_ridge = eta[0]
            best_mse_ridge = 1e10

            for i in eta:
                sdg = SDG(learning_rate=i, n_epochs=100, batch_size=10, method='ridge', lmbda= 0.01)
                beta = sdg.train(X_train, z_train)
                mse_ols_, mse_ridge_ = sdg.compute_test_mse(X_test, z_test, lambda_=0.01, beta=beta)
                MSE_ridge_val.append(mse_ridge_)
                if mse_ridge_ < best_mse_ridge:
                    best_beta_ridge = beta
                    best_mse_ridge = mse_ridge_

        if method == 'ols':

            eta = np.logspace(-5, -3, 10)
            best_beta_ols = eta[0]
            best_mse_ols = 1e10

            for i in eta:
                sdg = SDG(learning_rate=i, n_epochs=100, batch_size=10, method='ols', lmbda= 0.01)
                beta = sdg.train(X_train, z_train)
                mse_ols_, mse_ridge_ = sdg.compute_test_mse(X_test, z_test, lambda_=0.01, beta=beta)
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


plot_MSE_ols_ridge()