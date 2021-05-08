# Import modules
import numpy as np


def griewank(x):
    """Griewank's objective function.

    Has a global minimum of `0` at :code:`f(0,0,...,0)` with a search
    domain of [-600, 600]

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    ------
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not np.logical_and(x >= -600, x <= 600).all():
        raise ValueError("Input for Griewank function must be within [-600, 600].")

    d = x.shape[1]
    p = x.shape[0]

    i = np.tile(np.arange(start=1, stop=d + 1), reps=p).reshape(p, d)

    j = np.sum((x ** 2) / 4000, axis=1) - np.prod(np.cos(x / np.sqrt(i)), axis=1) + 1

    return j


def zakharov(x):
    """Zakharov's objective function.

    Has a global minimum of `0` at :code:`f(0,0,...,0)` with a search
    domain of [-600, 600]

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    ------
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not np.logical_and(x >= -5, x <= 10).all():
        raise ValueError("Input for Zakharov function must be within [-5, 10].")

    d = x.shape[1]
    p = x.shape[0]

    i = np.tile(np.arange(start=1, stop=d + 1), reps=p).reshape(p, d)

    j = np.sum(x ** 2, axis=1) + np.sum(0.5 * i * x, axis=1) ** 2 + np.sum(0.5 * i * x, axis=1) ** 4

    return j


def cigar(x):
    """Cigar's objective function.

    Has a global minimum of `0` at :code:`f(0,0,...,0)` with a search
    domain of [-100, 100]

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    ------
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not np.logical_and(x >= -100, x <= 100).all():
        raise ValueError("Input for Cigar function must be within [-100, 100].")

    j = x[:, 0] ** 2 + (10 ** 6) * np.sum(x[:, 1:] ** 2, axis=1)

    return j


def schwefel(x):
    """Schwefel's objective function.

    Has a global minimum of `0` at :code:`f(420.9687,420.9687,...,420.9687)` with a search
    domain of [-500, 500]

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    ------
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not np.logical_and(x >= -500, x <= 500).all():
        raise ValueError("Input for Schwefel function must be within [-500, 500].")

    d = x.shape[1]

    j = 418.9828872724339 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)

    return j


def salomon(x):
    """Salomon's objective function.

    Has a global minimum of `0` at :code:`f(0,0,...,0)` with a search
    domain of [-100, 100]

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    ------
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not np.logical_and(x >= -100, x <= 100).all():
        raise ValueError("Input for Salomon function must be within [-100, 100].")

    j = 1 - np.cos(2 * np.pi * np.sqrt(np.sum(x ** 2, axis=1))) + 0.1 * np.sqrt(np.sum(x ** 2, axis=1))

    return j
