from splitting_data import splitting_data
import numpy as np
from numpy.random import rand
from .fs_functions import init_position, binary_conversion, boundary, Fun


def gwo_feature_selection(dataframe, target_col, print_fs, k=5):
    X = dataframe.drop(columns=[target_col], axis=1).values
    y = dataframe[target_col].values

    xtrain, ytrain, xtest, ytest = splitting_data(X, y, train_size=0.8, require_val=False)
    fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

    opts = {
        'N': 50,       # Number of wolves
        'T': 100,      # Maximum number of iterations
        'k': k,        # K-value in KNN
        'fold': fold
    }

    selected_indices = gwo_jfs(xtrain, ytrain, opts, print_fs)['sf']
    feature_names = dataframe.drop(columns=[target_col]).columns

    selected_features = [feature_names[i] for i in selected_indices]

    return selected_features

def gwo_jfs(xtrain, ytrain, opts, print_fs):
    # Parameters
    ub    = 1
    lb    = 0
    thres = 0.5

    N        = opts['N']
    max_iter = opts['T']

    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position
    X      = init_position(lb, ub, N, dim)

    # Binary conversion
    Xbin   = binary_conversion(X, thres, N, dim)

    # Fitness at first iteration
    fit    = np.zeros([N, 1], dtype='float')
    Xalpha = np.zeros([1, dim], dtype='float')
    Xbeta  = np.zeros([1, dim], dtype='float')
    Xdelta = np.zeros([1, dim], dtype='float')
    Falpha = float('inf')
    Fbeta  = float('inf')
    Fdelta = float('inf')

    for i in range(N):
        fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts)
        if fit[i,0] < Falpha:
            Xalpha[0,:] = X[i,:]
            Falpha      = fit[i,0]

        if fit[i,0] < Fbeta and fit[i,0] > Falpha:
            Xbeta[0,:]  = X[i,:]
            Fbeta       = fit[i,0]

        if fit[i,0] < Fdelta and fit[i,0] > Fbeta and fit[i,0] > Falpha:
            Xdelta[0,:] = X[i,:]
            Fdelta      = fit[i,0]

    # Pre
    curve = np.zeros([1, max_iter], dtype='float')
    t     = 0

    curve[0,t] = Falpha.copy()
    if print_fs:
        print("Iteration:", t + 1)
        print("Best (HHO):", curve[0,t])
    t += 1

    while t < max_iter:
      	# Coefficient decreases linearly from 2 to 0
        a = 2 - t * (2 / max_iter)

        for i in range(N):
            for d in range(dim):
                # Parameter C (3.4)
                C1     = 2 * rand()
                C2     = 2 * rand()
                C3     = 2 * rand()
                # Compute Dalpha, Dbeta & Ddelta (3.5)
                Dalpha = abs(C1 * Xalpha[0,d] - X[i,d])
                Dbeta  = abs(C2 * Xbeta[0,d] - X[i,d])
                Ddelta = abs(C3 * Xdelta[0,d] - X[i,d])
                # Parameter A (3.3)
                A1     = 2 * a * rand() - a
                A2     = 2 * a * rand() - a
                A3     = 2 * a * rand() - a
                # Compute X1, X2 & X3 (3.6)
                X1     = Xalpha[0,d] - A1 * Dalpha
                X2     = Xbeta[0,d] - A2 * Dbeta
                X3     = Xdelta[0,d] - A3 * Ddelta
                # Update wolf (3.7)
                X[i,d] = (X1 + X2 + X3) / 3
                # Boundary
                X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])

        # Binary conversion
        Xbin  = binary_conversion(X, thres, N, dim)

        # Fitness
        for i in range(N):
            fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts)
            if fit[i,0] < Falpha:
                Xalpha[0,:] = X[i,:]
                Falpha      = fit[i,0]

            if fit[i,0] < Fbeta and fit[i,0] > Falpha:
                Xbeta[0,:]  = X[i,:]
                Fbeta       = fit[i,0]

            if fit[i,0] < Fdelta and fit[i,0] > Fbeta and fit[i,0] > Falpha:
                Xdelta[0,:] = X[i,:]
                Fdelta      = fit[i,0]

        curve[0,t] = Falpha.copy()
        if print_fs:
            print("Iteration:", t + 1)
            print("Best (HHO):", curve[0,t])
        t += 1


    # Best feature subset
    Gbin       = binary_conversion(Xalpha, thres, 1, dim)
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    gwo_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}

    return gwo_data