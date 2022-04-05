import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

from sklearn.model_selection import KFold

from outcome_models import stats

def data_wrapup(A, Y, U, nb_obs, model):

    data = preprocessing(A, Y, U, nb_obs)
    a, y, u = next(iter(data))

    _ , _, _, _, _, z_proxy = model(a.float(), y[:, :y.shape[1]-1])

    treatments = a.detach().numpy()
    treatments = (treatments + 1) /2.

    outcomes = y.detach().numpy()

    z_proxy = z_proxy.detach().numpy()

    z_proxy = (z_proxy - z_proxy.mean()) / z_proxy.std()

    random = np.random.uniform(-100, 100, a.shape[0])[None].T
    oracle = u.detach().numpy()[None].T

    random = oracle +  np.random.normal(0, .1, a.shape[0])[None].T

    data = {'outcomes':outcomes, 'treatments':treatments, 'proxy':z_proxy, 'random':random, 'oracle':oracle, 'none':None}

    return data


def kfold(dct, n_splits):

    X = np.array(dct['A']).T
    Y = np.array(dct['Y']).T
    U = np.array(dct['U']).T
    
    kf = KFold(n_splits=n_splits)
    kf.get_n_splits(X)

    KFold(n_splits=n_splits, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        U_train, U_test = U[train_index], U[test_index]
    
    return X_train, X_test, Y_train, Y_test, U_train,  U_test


def preprocessing(x, y, u, batch_size):
    data = []
    for k in np.arange(x.shape[0]):
        data.append([x[k], y[k], u[k]])

    data = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)
    return data


def select(tensor, position, order):

    x = tensor.numpy().T
    x = x[position:position+order, :]
    x = torch.from_numpy(x.T)

    return x


def softplus(tensor):

    m = nn.Softplus()

    return m(tensor)

def moving_average(curve, window):

    moving_average = []

    for idx in np.arange(len(curve)):
        up = min(idx + window, len(curve))
        down = max(idx - window, 0)
                
        moving_average.append(np.mean(curve[down:up]))

    return moving_average


def plot(pc_curve, pc_fit, lk_curve, lk_fit, ks_curve, ks_fit, boxe, save_dir):
    
    color_chart = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
    fig, ax1 = plt.subplots()
    
    pc_min = abs(np.array(pc_fit) - .5).argmin()
    lk_max = np.argmax(lk_fit)
    ks_max = np.argmax(ks_fit)


    #
    color = color_chart[0]
    ax1.plot(pc_curve, label='Surrogate PC', color=color, alpha=0.2)
    ax1.plot(pc_fit, label='Surrogate PC - Smoothed', color=color)
    ax1.plot(pc_min, pc_fit[pc_min], 'bo', color=color)

    color = color_chart[1]
    ax1.plot(lk_curve, label='Anova PC', alpha=0.2, color=color)
    ax1.plot(lk_fit, label='Anova PC - Smoothed', color=color)
    ax1.plot(lk_max, lk_fit[lk_max], 'bo', color=color)

    color = color_chart[2]
    ax1.plot(ks_curve, label='KS PC', alpha=0.2, color=color)
    ax1.plot(ks_fit, label='Anova PC - Smoothed', color=color)
    ax1.plot(ks_max, ks_fit[ks_max], 'bo', color=color)

    ax1.tick_params(axis ='y')

    # 
    ax2 = ax1.twinx()
    #ax2.plot(h_test, color='black', label='MSE - train', alpha=.6)

    #leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.125),
    #          fancybox=True, shadow=False, ncol=2)
    #leg.get_frame().set_alpha(0)

    ax1.set_ylabel('Predictive check (PC) likelihood', fontsize=10)
    ax2.set_ylabel('MSE', fontsize=10)
    plt.xlabel('Epoch', fontsize=10)

    ##
    k=0
    for texte in boxe:

        plt.text(.87, .3 - k, texte , horizontalalignment='center',
             verticalalignment='center', transform=ax2.transAxes)
        k+=.075
    
    plt.savefig(save_dir)


def mean_results(idx, models, A, Y, U, A_test, Y_test, U_test, nb_obs, strategies, start, order):
    H_train, H_test = [], []

    window = 1
    window_min, window_max = max(0, idx-window), min(len(models), idx+window)

    for k in range(window_min, window_max, 1):

        train_set = data_wrapup_2(A, Y, U, int(nb_obs), models[k])
        test_set = data_wrapup_2(A_test, Y_test, U_test, int(nb_obs), models[k])

        train = [train_set['outcomes'], train_set['treatments'], train_set['proxy']]
        test = [test_set['outcomes'], test_set['treatments'], test_set['proxy']]

        H_test.append(stats(train=train, test=test, strategies=strategies,
                            start=start, order=order, confounder_type='proxy')[0][0])

    return np.mean(H_test)