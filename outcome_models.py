import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression


def msm_long(outcomes, treatments, confounder, strategy, training=True, model_1=None, model_2=None):

    if training:
        model_1 = propensity_score(outcomes, treatments, confounder)

    weight, weight_stabilized = ip_weights(strategy, outcomes, treatments, confounder, model_1)

    #print(weight[:20])

    outputs, model_2 = recursive_model(strategy, outcomes, treatments, confounder, weight, weight_stabilized, model_2,
                                       training)

    return outputs, model_1, model_2


def propensity_score(outcomes, treatments, confounder):

    nb_treatments = treatments.shape[1]

    # Pooling data
    for k in np.arange(nb_treatments):
        if k == 0:
            if confounder is not None:
                data = np.concatenate((outcomes[:, :nb_treatments], confounder), axis=1)
            else:
                data = outcomes[:, :nb_treatments]
            target = treatments[:, k]
        else:
            if confounder is not None:
                x = np.concatenate((outcomes[:, k: k+nb_treatments], confounder), axis=1)
            else:
                x = outcomes[:, k: k+nb_treatments]

            data = np.concatenate((data, x), axis=0)
            target = np.concatenate((target, treatments[:, k]), axis=0)

    # Ps parameters
    clf = LogisticRegression(random_state=0, penalty='l2').fit(data, target)

    return clf


def ip_weights(strategy, outcomes, treatments, confounder, model):

    nb_treatments = treatments.shape[1]
    s, s_w = [], []

    # for every treatments
    for k in np.arange(nb_treatments):

        # create X
        if confounder is not None:
            x = np.concatenate((outcomes[:, k: k+nb_treatments], confounder), axis=1)
        else:
            x = outcomes[:, k: k+nb_treatments]

        # predictions
        prob = model.predict_proba(x)

        if k == 0:

            # revoir commentaire si c'est un ps ou un IPW
            s_w.append(prob[:, strategy[0]] ** -1)
            s.append(prob[:, 1] ** -1)
        else:
            s_w.append(s[-1] * prob[:, strategy[k]] ** -1)
            s.append(s[-1] * prob[:, 1]** -1)

    return np.array(s).T, np.array(s_w).T


def recursive_model(strategy, outcomes, treatments, confounder, weight, weight_stabilized, model=False, training=False):

    nb_obs, nb_treatments, order = treatments.shape[0], treatments.shape[1], treatments.shape[1]

    model_stack = []
    target = outcomes[:, -1][None].T
    for k in range(nb_treatments, 0, -1):

        # Construction
        floor = max(k - order, 0)
        if training:
            x = np.concatenate((outcomes[:, k-order+1: k+1], treatments[:, floor: k], weight[:, k-1][None].T), axis=1)
            if confounder is not None:
                x = np.concatenate((x, confounder), axis=1)

            # Fit
            reg = LinearRegression().fit(x, target)
            model_stack.append(reg)

        # Construction
        a = (np.ones(nb_obs) * strategy[k-1])[None].T
        x = np.concatenate((outcomes[:, k-order+1: k+1], treatments[:, floor: k-1], a, weight_stabilized[:,
                                                                                         k-1][None].T), axis=1)

        if confounder is not None:
            x = np.concatenate((x, confounder), axis=1)

        if training:
            target = reg.predict(x)
        else:
            target = model[k-1].predict(x)

    return target, model_stack


def stats(train, test, strategies):

    outcomes_train, treatments_train, confounder_train = train
    outcomes_test, treatments_test, confounder_test = test

    num_treatments = treatments_test.shape[1]

    mse = 0
    for strategy in strategies:

        # Training the outcome model
        _, model_1, model_2 = msm_long(outcomes_train, treatments_train, confounder_train, strategy)

        #print(model_2[-1].coef_)

        # Prediction on the test
        predictions, _, _ = msm_long(outcomes_test, treatments_test, confounder_test, strategy,
                                           training=False, model_1=model_1, model_2=np.flip(model_2))

        # Number of observation for this strategy
        dummy = (abs(treatments_test[:, :num_treatments] - strategy).sum(axis=1) == 0)[None].T
        # Std MSE
        diff = np.sqrt((predictions - outcomes_test[:, -1][None].T) ** 2)
        # Only consider the counterfactual outcome who match with the selected strategy
        mse += np.dot(diff.T, dummy)

    return mse / len(dummy)
