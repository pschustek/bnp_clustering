# -*- coding: utf-8 -*-
"""
Generate observations
"""
import numpy as np

# Means of Gaussians
mu = np.array([[0.8, -0.8], [-1, 0.4],
               [0.7, np.sqrt(1-0.7**2)] + np.array([2.2, 0.7])*0.5])

# Covariance matrices
w1 = np.array([[1, 0.5], [0.5, 2]]) * 0.2
w2 = np.array([[2, 0.5], [0.5, 1]]) * 0.45
w3 = np.array([[2, -1], [-1, 1]]) * 0.1
W = np.array([w1, w2, w3])

# Mixture coefficients
mixture_coeff = np.array([0.3, 0.5, 0.2])


def sample(numSample=300):
    """
    Sample from Gaussian mixture model
    """
    component = np.random.multinomial(1, mixture_coeff, numSample)
    data = []
    for j, cov in enumerate(W):
        data.append(np.random.multivariate_normal(mu[j, :], W[j, :, :], np.sum(component, axis=0)[j]))

    return data


def balanced_subsampling(sample, xlb, xub, ylb, yub, num=60):
    """
    Component-wise, approximately balanced subsampling for data from mixture model
    """

    def assign_remaining(num, num_sector):
        """
        Integer division that approximate proportions of sample
        """
        if num == 1:
            num_sector.append(number_per_component[c] - np.sum(num_sector))
        else:
            num_sector.append(np.round((number_per_component[c] - np.sum(num_sector)) / num))
            assign_remaining(num - 1, num_sector)

    def bin_mask(x, nbins):
        """
        Bins data vector based on cumulative quantiles and returns logical mask
        """
        qtls = np.arange(nbins+1)/nbins
        # Bin edges through cum. quantiles
        edges = np.percentile(x, qtls * 100)
        # Bin data points
        idx = np.digitize(x, edges)
        # Mask array for each bin (column)
        return np.equal(idx[:, np.newaxis], range(1, nbins+1))

    # TODO more general solution
    number_per_component = [np.round(mixture_coeff[0]*num), np.floor(mixture_coeff[2]*num)]
    number_per_component.insert(1, num - np.sum(number_per_component))

    subsampled_data = []    # component-wise
    for c, D in enumerate(sample):
        # Only subsample visible values within axes limits
        mask = np.logical_and(np.logical_and(D[:, 0] > xlb, D[:, 0] < xub), np.logical_and(D[:, 1] > ylb, D[:, 1] < yub))
        dat = D[mask, :]

        # Divide into quadratic grid from which a roughly equal number of points is sampled
        nbins = 2

        # Determine number of points for grid sectors
        num_grid = nbins**2

        # Write in original list num_sector with recursive function
        num_sector = [np.round(number_per_component[c]/num_grid)]
        assign_remaining(num_grid - 1, num_sector)

        mask_x = bin_mask(dat[:, 0], nbins)
        mask_y = bin_mask(dat[:, 1], nbins)

        j = 0
        S = []
        for ix in range(nbins):
            for iy in range(nbins):
                mask = np.logical_and(mask_x[:, ix], mask_y[:, iy])
                idx = np.random.choice(np.arange(np.sum(mask)), int(num_sector[j]), replace=False)
                subsample = dat[mask,:][idx]
                j += 1
                S.append(subsample)

        S = np.vstack(S)
        subsampled_data.append(S)

#        plt.scatter(sample[c][:,0], sample[c][:,1])
#        plt.scatter(S[:,0], S[:,1])

    return np.vstack(subsampled_data), S