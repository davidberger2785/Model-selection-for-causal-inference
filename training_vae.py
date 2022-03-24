import copy
import numpy as np


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
    kl_train_stack, kl_valid_stack = [], []
    recon_train_stack, recon_valid_stack = [], []
    err_train, err_valid = [], []

    for step in np.arange(nb_epoch):
        model_stack.append(copy.deepcopy(model))

        ## Fixer en fonction du commentaire d'allac!
        if step > model.kl_begin & step < model.kl_end:
            model.kl_coef += 0.001

        train = True
        for Set in [train_set, valid_set]:

            loss_compil, recon_compil, kl_compil, err, count = .0, .0, .0, 0., 0.
            for a, y, _ in Set:

                if train:
                    optimizer.zero_grad()

                z_mean_q, z_logvar_q, z_mean_p, z_logvar_p, a_hat, z_sample = model(a.float(), y)
                loss, recon_loss, kl = model.loss(a.float(), a_hat, z_mean_q, z_logvar_q, z_mean_p, z_logvar_p)

                loss_compil += loss * a.shape[0]
                recon_compil += recon_loss * a.shape[0]
                kl_compil += kl * a.shape[0]

                count += a.shape[0] * a.shape[1]

                err += ((a_hat > 0.5)*1 != a).sum()

                if train:
                    loss.backward()
                    optimizer.step()

            if train:
                loss_train_stack.append(loss_compil / count)
                recon_train_stack.append(recon_compil / count)
                kl_train_stack.append(kl_compil / count)
                err_train.append(err/(count * a.shape[1]))
            else:
                loss_valid_stack.append(loss_compil / count)
                recon_valid_stack.append(recon_compil / count)
                kl_valid_stack.append(kl_compil / count)
                err_valid.append(err/(count * a.shape[1]))

            train = False

    return model_stack, loss_train_stack, loss_valid_stack, recon_train_stack, recon_valid_stack, kl_train_stack, kl_valid_stack, err_train, err_valid
