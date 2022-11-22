import numpy as np


def train_test_split(X, y, test_size=0.2):
    assert len(X) == len(y)
    x_len = len(X)
    samp_size = int(x_len*test_size)
    test_idx = np.random.choice(
        range(len(X)), size=samp_size, replace=False
    )
    train_idx = list(
        set(range(x_len)) - set(test_idx)
    )
    Xtrain = [X[i] for i in train_idx]
    ytrain = [y[i] for i in train_idx]
    Xtest = [X[i] for i in test_idx]
    ytest = [y[i] for i in test_idx]
    print('***train/test set created of length:{}/{}'.format(
        len(Xtrain), len(Xtest)))
    return Xtrain, Xtest, ytrain, ytest
