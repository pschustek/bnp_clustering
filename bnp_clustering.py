# -*- coding: utf-8 -*-
"""
Bayesian nonparametric clustering

This is a test bench for variational inference of a Gaussian mixture model.

Note: Needs to be extended to work for more than two dimensions
"""
import settings as sg
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.special import digamma as psi
from scipy.stats import multivariate_normal as mvn
#import pdb


def inference_vgmm(X, num_comp, sigma, nu0=200, beta0=1, alpha0=1):
    """
    BNP clustering implemented by variational mixture model of Gaussians

    Symmetry and a zero-mean prior of the components is assumed.
    The calculations need to be generalized to work for dimension greater than two.
    The arguments 3-6 define the prior distribution over the latent variables.

    Arguments:
        X: array (n x d)
            Input data of length n and dimensionality d.
        num_comp:
            Number of components of the mixture to initialize the algorithm.
            They will effectively get pruned if they are not useful to capture the data.
        sigma:
            Used to define isotropic prior distribution over the precision matrix
        nu:
            Degree of freedom parameter of the Wishart prior distribution over the precision matrix
        beta0:
            Scaling parameter of the precision parameter of the Gaussian prior distribution over the component means
        alpha0:
            Parameters of the Dirichlet prior distribution over the mixture coefficients
    """

    # K-means to get initial values for the means
    k_means = KMeans(num_comp, init='random', n_init=20)
    k_means.fit(X)
    mk = k_means.cluster_centers_

    # dimensionality
    dim = X.shape[1]

    # Prior distributions:
    # Wishart (precision)
    # Set E[precision] = 1/sigma^2 = W0*nu0
    W0 = np.eye(dim) / (sigma**2 * nu0)

    # Zero mean of Gaussian-Wishart
    m0 = np.zeros(dim)

    # Symmetric Dirichlet
    alpha0 = np.ones(num_comp) * alpha0

    # Initialize all parameters
    Wk = np.tile(W0 / 1.5, (num_comp, 1, 1))
    nuk = np.tile(nu0, num_comp)
    betak = np.tile(beta0, num_comp)
    alphak = alpha0

    x = X[:, 0]
    y = X[:, 1]

    # E-step
    def e_step():
        """
        Update the responsibilities
        """

        def raw_responsibility_k(alphak, alpha, Wk, mk, nuk, betak):
            lambda_k = np.exp(np.sum(psi((nuk + 1 - np.arange(1, dim + 1)) / 2))
                + dim*np.log(2) + np.log(np.linalg.det(Wk)))

            pi_k = np.exp(psi(alphak) - psi(np.sum(alpha)))

            rho_n = np.zeros(len(x)) * np.nan

            for n in range(len(x)):
                rho_n[n] = pi_k * np.sqrt(lambda_k) * np.exp( -dim / (2 * betak) - nuk / 2 * np.dot(np.dot((X[n, :] - mk), Wk), (X[n,:] - mk)))

            return rho_n

        # Component-wise for all points
        rho_nk = np.zeros((len(x), num_comp)) * np.nan
        for k in range(num_comp):
            rho_nk[:, k] = raw_responsibility_k(alphak[k], alphak, Wk[k, :, :], mk[k, :], nuk[k], betak[k])

        # Renormalize
        norm = np.sum(rho_nk, axis=1)
        r_nk = np.divide(rho_nk, norm[:, np.newaxis])
        return r_nk

    # M-step
    def m_step():
        """
        Estimate all parameters
        """
        # Useful quantities
        Nk = np.sum(r_nk, 0)

        xmk = np.stack((np.divide(np.sum(r_nk * x[:, np.newaxis], axis=0), Nk),      # TODO generalize
                        np.divide(np.sum(r_nk * y[:, np.newaxis], axis=0), Nk)))
        Sk = []
        for k in range(num_comp):
            mat = np.zeros((dim, dim))
            dfr = X - xmk[:, k]

            for n in range(X.shape[0]):
                mat += r_nk[n, k] * np.outer(dfr[n], dfr[n])

            Sk.append(mat/Nk[k])

        # M-step updating
        alphak = alpha0 + Nk

        betak = beta0 + Nk

        mk = np.divide(beta0 * m0[:, np.newaxis] + xmk * Nk, betak).T

        Wk = []
        for k in range(num_comp):
            dfr = xmk[:,k] - m0
            Wkinv = np.linalg.inv(W0) + Nk[k] * Sk[k] + np.outer(dfr, dfr) * beta0 * Nk[k] / (beta0+Nk[k])
            Wk.append(np.linalg.inv(Wkinv))

        nuk = nu0 + Nk

        return alphak, betak, mk, np.stack(Wk), nuk

    # EM-Algorithm
    # Cycle between E and M step until convergence
    old = np.ones((X.shape[0], num_comp))    # for responsibilities
    iterate = 1
    j = 0
    V = np.zeros(100) * np.nan     # preallocate

    while iterate:
        # E-step
        r_nk = e_step()

        if np.sum(np.isnan(r_nk)):
            raise ValueError('Some responsibilities r_nk are NaN', 'r_nk')

        # M-step
        alphak, betak, mk, Wk, nuk = m_step()

        # Monitor changes
        V[j] = np.sum(np.mean(np.divide(np.abs(r_nk - old), np.abs(old) + 1e-20), 0))
        # Average over recent measures for robustness
        measure = np.mean(V[np.arange(np.amax([0, j-2]), j+1)])
        if np.logical_and(j > 1, measure < 1e-8):
            iterate = 0

        old = r_nk

        if np.mod(j+1, 100) == 0:
            V = np.append(V, np.zeros(100) * np.nan)

        j += 1

    # Compute some expectations
    coeff = alphak / np.sum(alphak)
    Ck = []
    for k in range(num_comp):
        # Expectation of corresponding inverse Wishart distribution
        Ck.append(np.linalg.inv(Wk[k, :, :] * nuk[k]))

    return mk, np.stack(Ck), coeff, alphak, betak, nuk, r_nk


def draw_cov_ellipse(Ck, mk, coeff=1, line_color='red', lw=1, num_points=200):
    """
    Draw ellipse based on covariance matrix Ck
    """

    # Columns correspond to eigen values
    eigen_val, eigen_vec = np.linalg.eig(Ck)

    # Angle between x-axis and largest semimajor axis
    idx = np.argsort(eigen_val)
    rad = np.arctan2(eigen_vec[1, idx[1]], eigen_vec[0, idx[1]])    # y/x

    # This angle is between -pi and pi.
    # Let's shift it such that the angle is between 0 and 2pi
    # Close the circle and convert to polar axis
    # (-pi and pi refer to the same directions)
    if rad < 0:
        rad += 2 * np.pi

    t = np.linspace(0, 2*np.pi, num_points)
    X = np.sqrt(eigen_val[idx[1]]) * np.cos(t)
    Y = np.sqrt(eigen_val[idx[0]]) * np.sin(t)

    # Define a rotation matrix
    R = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
    # Rotate the ellipse
    r_ellipse = np.dot(np.stack((X, Y), 1), R)

    lwidth = 0.3*(1-coeff) + lw*coeff

    # Plot
    plt.plot(r_ellipse[:, 0] + mk[0], r_ellipse[:,1] + mk[1], 1, color=line_color,
             linewidth=lwidth)


def pdf_vgmm(mk, Ck, p, gvx, gvy):
    """
    Plot density of Gaussian mixture model for given point estimates
    """
    gX, gY = np.meshgrid(gvx, gvy)

    xygrid = np.empty(gX.shape + (2,))
    xygrid[:, :, 0] = gX
    xygrid[:, :, 1] = gY

    # Loop over output grid points
    pdf = np.zeros(gX.shape)
    for k in range(len(p)):
        pdf += mvn.pdf(xygrid, mk[k], Ck[k]) * p[k]

    return pdf, gX, gY

# %% Calculate

# For reproducible results
np.random.seed(3571237)
subsample = 0

# Define bounds for plotting
xlb = -2.5
xub = 2.5
ylb = -2
yub = 2

# List with component-wise samples
all_data = sg.sample(80)

# Optional: A representative smaller sample (for visualization)
if subsample:
    data, component_data = sg.balanced_subsampling(all_data, xlb, xub, ylb, yub)
else:
    data = np.concatenate(all_data, 0)

# Adjust parameters of inference algorithm here
mk, Ck, p, *misc = inference_vgmm(data, 6, 0.6)

# %% Plot
# Color definitions
color_sample = [0.1, 0.8, 0]        # to illustrate sample
color_estimated = [0.9, 0.7, 0]       # estimated clusters

gvx = np.linspace(xlb, xub, 50)
gvy = np.linspace(ylb, yub, 50)

pdf, gX, gY = pdf_vgmm(mk, Ck, p, gvx, gvy)
cmap = plt.cm.gray.reversed()
plt.contourf(gX, gY, pdf, 20, cmap=cmap)

plt.scatter(data[:, 0], data[:, 1], 12, c=color_sample)

for k in range(sg.mu.shape[0]):
    draw_cov_ellipse(sg.W[k, :, :], sg.mu[k, :], sg.mixture_coeff[k], color_sample, 8)

for k in range(Ck.shape[0]):
    draw_cov_ellipse(Ck[k, :, :], mk[k, :], p[k], color_estimated, 8)

plt.xlim(xlb, xub)
plt.ylim(ylb, yub)
