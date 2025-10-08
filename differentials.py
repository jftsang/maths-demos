import numpy as np
from numpy import pi
from numpy import sqrt, sin, cos, tan, sinh, cosh, exp, dot
from scipy.linalg import eig, eigh


def discretise(xa, xb, nx=101, periodic=True):
    """Returns the xs, the spacing dx between the xs,
    and the identity and first and second derivative
    matrices, computed using a 2nd-order scheme."""
    xs = np.linspace(xa, xb, nx)
    dx = xs[1] - xs[0]

    eye = np.eye(nx)

    # Use central finite differences
    ddx = (np.roll(eye, 1, 1) - np.roll(eye, -1, 1)) / (2*dx)
    d2dx2 = (np.roll(eye, 1, 1) - 2*eye + np.roll(eye, -1, 1)) / (dx*dx)

    if periodic:
        pass
    else:
        ddx[0, 4:] = 0
        ddx[0, :4] = (-11/6, 3, -3/2, 1/3) / dx
        ddx[-1, :-4] = 0
        ddx[-1, -4:] = (-1/3, 3/2, -3, 11/6) / dx

        d2dx2[0, 4:] = 0
        d2dx2[0, :4] = (2, -5, 4, -1) / (dx*dx)
        d2dx2[-1, :-4] = 0
        d2dx2[-1, -4:] = (-1, 4, -5, 2) / (dx*dx)

    return xs, dx, eye, ddx, d2dx2


def d4dx4_mat(nx, dx=1, periodic=True):
    eye = np.eye(nx)
    d4dx4 = (np.roll(eye, 2, 1) - 4*np.roll(eye, 1, 1) + 6*eye
                 - 4*np.roll(eye, -1, 1) + np.roll(eye, -2, 1)) / dx**4

    if periodic:
        pass
    else:
        d4dx4[0, :] = 0;
        d4dx4[0, :5] = (1, -4, 6, -4, 1) / dx**4
        d4dx4[1, :] = 0
        d4dx4[1, 1:6] = (1, -4, 6, -4, 1) / dx**4;
        d4dx4[-2, :] = 0;
        d4dx4[-2, -6:-1] = (1, -4, 6, -4, 1) / dx**4
        d4dx4[-1, :] = 0
        d4dx4[-1, -5:] = (1, -4, 6, -4, 1) / dx**4;

    return d4dx4


def d4dx4_mat(nx, dx=1):
    eye = np.eye(nx)
    d4dx4 = (np.roll(eye, 2, 1) - 4*np.roll(eye, 1, 1) + 6*eye
                 - 4*np.roll(eye, -1, 1) + np.roll(eye, -2, 1)) / dx**4
    return d4dx4


def myeig(op, eye, *args, **kwargs):
    """Wrapper around scipy.linalg.eig that returns eigenvalues
    in order of their real parts, and normalizes the eigenvectors.
    """
    nx = op.shape[0]
    eigvals, eigvecs = eig(op, eye, *args, **kwargs)
    ordering = sorted(range(nx), key=lambda i: np.real(eigvals)[i])
    eigvals = eigvals[ordering]
    eigvecs = eigvecs[:, ordering] * sqrt(nx)
    return eigvals, eigvecs
