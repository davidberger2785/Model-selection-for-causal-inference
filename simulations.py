import numpy as np
from scipy.special import expit


def outcomes_par(p, seed=False):
    """
    Generated weights of the outcome SCM with respect to AR(P) process

    Parameters:
    -----------
    p: int
        The order of the autoregressive model
    seed: int

    Returns:
    --------
    t-p, ..., t-1
        alpha and beta parameters as described in Section 7.1

    """
    #if seed != bool:
    #    np.random.seed(seed)

    k = np.arange(0, p)
    alpha = np.random.normal(1 - k/p, p ** -2)[::-1]
    beta = np.random.normal(1- k/p, p ** -2)[::-1]
    return alpha, beta


def treatments_par(p, treatment_effect=False, seed=False):
    """
    Generate weights of the treatment SCM with respect to AR(P) process

    Parameters:
    -----------
    p: int
        The order of the autoregressive model
    treatment_effect: bool
        Are past treatments taken into account to define future treatments

    seed: int

    Returns:
    --------
    delta: array
        p parameters as described in Section 7.1

    """
    #if seed != bool:
    #    np.random.seed(seed)

    k = np.arange(0, p)
    delta = np.random.normal(1 - k/p, p ** -2)[::-1]

    if treatment_effect:
        omega = np.random.normal(1 - k/p, p ** -2)[::-1]
        return delta, omega

    return delta


def AR(nb_obs, seq_length, order, mu_noise, sigma_noise, theta_y, theta_a,
       confounder_distribution, mu_confounder=0, sigma_confounder=1, poisson_par = 2,
       treatment_effect=False, seed=False):
    """
    Generate AR(order) process according to Section 7.1 of Berger et al. 2021.

    Parameters:
    -----------
    nb_obs: int
        Number of observations
    seq_length: int
        Length of the sequence (every observation have the same sequence length)
    order: int
        The order of the autoregressive process
    mu_noise: float
        Noise follows a Gaussian distribution with mean equals mu_noise
    sigma_noise: float
        Noise follows a Gaussian distribution with variance equals mu_noise
    confounder_distribution: string
        Distribution type of the confounder (binomial or normal)
    mu_confounder: float
        The mean if confounder follows a Gaussian distribution
    sigma_confounder: float
        The variance if confounder follows a Gaussian distribution
    treatment_effect: bool
        Are past treatments taken into account to define future treatments
    seed: int


    Returns:
    --------
    A: list
        Treatments
    Y: list
        Outcomes
    P: list
        Parameter of the bernouilli distribution associated to treatments process
    U: list
        Confounder

    """

    # data structures
    Y, A, P = [], [], []

    # VAR parameters
    alpha, beta = outcomes_par(order)

    #alpha = 1* np.array([1, 1])
    #beta = 1* np.array([1, 1])


    if treatment_effect:
        delta, omega = treatments_par(order, treatment_effect, seed)
    else:
        delta = treatments_par(order, treatment_effect, seed)

    #delta = 1*np.array([1, 1])

    # Confounders
    if confounder_distribution == 'normal':
        U = np.random.normal(mu_confounder, sigma_confounder, nb_obs)
    if confounder_distribution == 'binomial':
        U = 2*np.random.binomial(1, .5, nb_obs) - 1
    if confounder_distribution == 'poisson':
        U = np.random.poisson(poisson_par, nb_obs)

    # Generate sequence
    for k in np.arange(seq_length):
        if k == 0:

            # outcomes
            #y = theta_y * U + np.random.normal(mu_noise, sigma_noise, nb_obs)
            y = np.random.normal(mu_noise, sigma_noise, nb_obs)**2

            # treatments
            p = expit(theta_a * U + (1-theta_a) * delta[0] * y)
            a = 2 * np.random.binomial(1, p, nb_obs) - 1

            Y.append(y), A.append(a), P.append(p)

        else:
            j = min(k, order)

            # Outcomes
            Y.append(
                theta_y * U**2 + \
                (1 - theta_y) * (alpha[-j:] @ Y[-j:] + beta[-j:] @ A[-j:]) / j + \
                np.random.normal(mu_noise, sigma_noise, nb_obs)
            )

            # Treatments
            if treatment_effect:
                P.append(
                expit(
                theta_a * U + \
                (1 - theta_a) * (delta[-j:] @ Y[-j:] + omega[-j:] @ A[-j:]) / j
                ))
            else:
                P.append(
                expit(
                theta_a * U + \
                (1 - theta_a) * (delta[-j:] @ Y[-j:]) / j
                ))

            A.append(2 * np.random.binomial(1, P[-1], nb_obs) - 1)

    #return A, Y, P, U, {"alpha":alpha, "beta":beta, "delta":delta}
    return A, Y, P, U