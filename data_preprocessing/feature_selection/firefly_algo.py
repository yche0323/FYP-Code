from splitting_data import splitting_data
import numpy as np
from numpy.random import rand
from .fs_functions import Fun, init_position, boundary, binary_conversion


def fa_feature_selection(dataframe, target_col, print_fs, k=5):
    X = dataframe.drop(columns=[target_col], axis=1).values
    y = dataframe[target_col].values

    xtrain, ytrain, xtest, ytest = splitting_data(X, y, train_size=0.8, require_val=False)
    fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

    opts = {
        'N': 50,       # Number of fireflies
        'T': 100,      # Maximum number of iterations
        'alpha': 1,    # Alpha coefficient
        'beta0': 1,    # Initial attractiveness
        'gamma': 1,    # Absorption coefficient
        'theta': 0.97, # Control alpha
        'k': k,        # k-value in k-nearest neighbour
        'fold': fold
    }

    selected_indices = fa_jfs(xtrain, ytrain, opts, print_fs)['sf']
    feature_names = dataframe.drop(columns=[target_col]).columns

    selected_features = [feature_names[i] for i in selected_indices]

    return selected_features

def fa_jfs(xtrain, ytrain, opts, print_fs):
    # Parameters
    ub     = 1
    lb     = 0
    thres  = 0.5
    alpha  = 1       # constant
    beta0  = 1       # light amplitude
    gamma  = 1       # absorbtion coefficient
    theta  = 0.97    # control alpha

    N          = opts['N']
    max_iter   = opts['T']
    if 'alpha' in opts:
        alpha  = opts['alpha']
    if 'beta0' in opts:
        beta0  = opts['beta0']
    if 'gamma' in opts:
        gamma  = opts['gamma']
    if 'theta' in opts:
        theta  = opts['theta']

    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position
    X     = init_position(lb, ub, N, dim)

    # Binary conversion
    Xbin  = binary_conversion(X, thres, N, dim)

    # Fitness at first iteration
    fit   = np.zeros([N, 1], dtype='float')
    Xgb   = np.zeros([1, dim], dtype='float')
    fitG  = float('inf')

    for i in range(N):
        fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts)
        if fit[i,0] < fitG:
            Xgb[0,:] = X[i,:]
            fitG     = fit[i,0]

    # Pre
    curve = np.zeros([1, max_iter], dtype='float')
    t     = 0

    curve[0,t] = fitG.copy()
    if print_fs:
        print("Iteration:", t + 1)
        print("Best (HHO):", curve[0,t])
    t += 1

    while t < max_iter:
        # Alpha update
        alpha = alpha * theta
        # Rank firefly based on their light intensity
        ind   = np.argsort(fit, axis=0)
        FF    = fit.copy()
        XX    = X.copy()
        for i in range(N):
            fit[i,0] = FF[ind[i,0]]
            X[i,:]   = XX[ind[i,0],:]

        for i in range(N):
            # The attractiveness parameter
            for j in range(N):
                # Update moves if firefly j brighter than firefly i
                if fit[i,0] > fit[j,0]:
                    # Compute Euclidean distance
                    r    = np.sqrt(np.sum((X[i,:] - X[j,:]) ** 2))
                    # Beta (2)
                    beta = beta0 * np.exp(-gamma * r ** 2)
                    for d in range(dim):
                        # Update position (3)
                        eps    = rand() - 0.5
                        X[i,d] = X[i,d] + beta * (X[j,d] - X[i,d]) + alpha * eps
                        # Boundary
                        X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])

                    # Binary conversion
                    temp      = np.zeros([1, dim], dtype='float')
                    temp[0,:] = X[i,:]
                    Xbin      = binary_conversion(temp, thres, 1, dim)

                    # fitness
                    fit[i,0]  = Fun(xtrain, ytrain, Xbin[0,:], opts)

                    # best update
                    if fit[i,0] < fitG:
                        Xgb[0,:] = X[i,:]
                        fitG     = fit[i,0]

        # Store result
        curve[0,t] = fitG.copy()
        if print_fs:
            print("Iteration:", t + 1)
            print("Best (HHO):", curve[0,t])
        t += 1


    # Best feature subset
    Gbin       = binary_conversion(Xgb, thres, 1, dim)
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    fa_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}

    return fa_data