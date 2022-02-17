import numpy as np
import copy

def training(train_set, valid_set, model, optimizer, nb_epoch):
    """
     Decription

     Parameters:
     -----------

     Returns:
     --------


     """

    model_stack = []
    loss_train_stack, loss_valid_stack = [], []

    for step in np.arange(nb_epoch):

        model_stack.append(copy.deepcopy(model))

        train = True
        for Set in [train_set, valid_set]:

            loss_compil, count = .0, 0
            for a, y, _ in Set:

                if train:
                    optimizer.zero_grad()

                z_mean_q, z_logvar_q, a_hat, z_sample = model(a.float())
                loss, llk, kl = model.loss(a.float(), a_hat, z_mean_q, z_logvar_q)

                loss_compil += loss * a.shape[0]
                count += a.shape[0]

                if train:
                    loss.backward()
                    optimizer.step()

            if train:
                loss_train_stack.append(loss_compil/count)
            else:
                loss_valid_stack.append(loss_compil/count)

            train = False

    return model_stack, loss_train_stack, loss_valid_stack
