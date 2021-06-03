"""
Provides a random set of utility methods.

.. include:: ../include/links.rst
"""
import warnings

from IPython import embed

import numpy as np
from scipy import sparse, linalg, stats, special, ndimage

from astropy.stats import sigma_clip

# TODO: Add a set of weights?
def get_map_bin_transformations(spatial_shape=None, binid=None):
    r"""
    Construct various arrays used to convert back and forth between a 2D map
    and the associated vector of (unique) binned quantities.

    The input is expected to be for 2D maps with a given "spatial shape". For
    the method to run, you need to provide one of the two arguments;
    precedence is given to ``binid``.

    Provided an independent calculation of the value in each map position,
    this method provides the transformation matrix, :math:`\mathbf{T}`, used
    to calculate the binned values:
    
    .. math::

        \mathbf{b} = \mathbf{T} \dot \mathbf{m},

    where :math:`\mathbf{b}` is the vector with the binned data and
    :math:`\mathbf{m}` is the vector with the flattened map data.

    If all spaxels are independent, :math:`\mathbf{T}` in the above operation
    simply (down)selects, and possibly reorders, elements in
    :math:`\mathbf{m}` to match the bin indices.

    Parameters
    ----------
    spatial_shape : :obj:`tuple`, optional
        The 2D spatial shape of the mapped data. Ignored if ``binid`` is
        provided.

    binid : `numpy.ndarray`_, optional
        The 2D array providing the 0-indexed bin ID number associated with
        each map element. Bin IDs of -1 are assumed to be ignored; no bin ID
        can be less than -1. Shape is ``spatial_shape`` and its size (i.e.
        the number of grid points in the map) is :math:`N_{\rm spaxel}`.

    Returns
    -------

    ubinid : `numpy.ndarray`_
        1D vector with the sorted list of unique bin IDs. Shape is
        :math:`(N_{\rm bin},)`. If ``binid`` is not provided, this is
        returned as None.

    nbin : `numpy.ndarray`_
        1D vector with the number of spaxels in each bin. Shape is
        :math:`(N_{\rm bin},)`. If ``binid`` is not provided, this is just a
        vector of ones. The number of bins can also be determined from the
        returned ``bin_transform`` array::

            assert np.array_equal(nbin, np.squeeze(np.asarray(np.sum(bin_transform > 0, axis=1)))))

    ubin_indx : `numpy.ndarray`_
        The index vector used to select the unique bin values from a
        flattened map of binned data, *excluding* any element with ``binid ==
        -1``. Shape is :math:`(N_{\rm bin},)`. If ``binid`` is not provided,
        this is identical to ``grid_indx``. These indices can be used to
        reconstruct the list of unique bins; i.e.::

            assert np.array_equal(ubinid, binid.flat[ubin_indx])

    grid_indx : `numpy.ndarray`_
        The index vector used to select valid grid cells in the input maps;
        i.e., any grid point with a valid bin ID (``binid != -1``). Shape is
        :math:`(N_{\rm valid},)`. For example::

            indx = binid > -1
            assert np.array_equal(binid[indx], binid[np.unravel_index(grid_indx, binid.shape)])

    bin_inverse : `numpy.ndarray`_
        The index vector applied to a recover the mapped data given the
        unique quantities, when used in combination with ``grid_indx``. Shape
        is :math:`(N_{\rm valid},)`. For example::

            _binid = np.full(binid.shape, -1, dtype=int)
            _binid[np.unravel_index(grid_indx, binid.shape)] = ubinid[bin_inverse]
            assert np.array_equal(binid, _binid)
        
    bin_transform : `scipy.sparse.csr_matrix`_
        A sparse matrix that can be used to construct the binned set of
        quantities from a full 2D map. See :math:`\mathbf{T}` in the method
        description. Shape is :math:`(N_{\rm bin}, N_{\rm spaxel})`. Without
        any weighting, :math:`\mathbf{T}` just constructs the average of the
        values within the map that is applied to. In this case (or if all of
        the bins only contain a single spaxel), the following should pass::

            assert np.array_equal(ubinid, bin_transform.dot(binid.ravel()).astype(int))

    """
    if spatial_shape is None and binid is None:
        raise ValueError('Must provide spatial_shape or binid')
    _spatial_shape = spatial_shape if binid is None else binid.shape

    nspax = np.prod(_spatial_shape)
    grid_indx = np.arange(nspax, dtype=int)
    if binid is None:
        # All bins are valid and considered unique
        bin_transform = sparse.coo_matrix((np.ones(np.prod(spatial_shape), dtype=float),
                                           (grid_indx,grid_indx)),
                                          shape=(np.prod(spatial_shape),)*2).tocsr()
        return None, np.ones(nspax, dtype=int), grid_indx.copy(), grid_indx, grid_indx.copy(), \
                bin_transform

    # Get the indices of measurements with unique bin IDs, ignoring any
    # IDs set to -1
    binid_map = binid.ravel()
    ubinid, ubin_indx, bin_inverse, nbin \
            = np.unique(binid_map, return_index=True, return_inverse=True, return_counts=True)
    if np.any(ubinid == -1):
        ubinid = ubinid[1:]
        ubin_indx = ubin_indx[1:]
        grid_indx = grid_indx[bin_inverse > 0]
        bin_inverse = bin_inverse[bin_inverse > 0] - 1
        nbin = nbin[1:]

    # NOTE: In most cases, ubinid[bin_inverse] is identical to bin_inverse. The
    # exception is if the bin numbers are not sequential, i.e., the bin numbers
    # are not identical to np.arange(nbin).

    # Construct the bin transform using a sparse matrix
    d,i,j = np.array([[1/nbin[i],i,j] 
                     for i,b in enumerate(ubinid)
                     for j in np.where(binid_map == b)[0]]).T
    bin_transform = sparse.coo_matrix((d,(i.astype(int),j.astype(int))),
                                      shape=(ubinid.size, np.prod(_spatial_shape))).tocsr()

    return ubinid, nbin, ubin_indx, grid_indx, bin_inverse, bin_transform


def impose_positive_definite(mat, min_eigenvalue=1e-10, renormalize=True):
    """
    Force a matrix to be positive definite.

    Following, e.g.,
    http://comisef.wikidot.com/tutorial:repairingcorrelation, the algorithm
    is as follows:

        - Calculate the eigenvalues and eigenvectors of the provided matrix
          (this is the most expensive step).
        - Impose a minimum eigenvalue (see ``min_eigenvalue``)
        - Reconstruct the input matrix using the eigenvectors and the
          adjusted eigenvalues
        - Renormalize the reconstructed matrix such its diagonal is identical
          to the input matrix, if requested.

    Args:
        mat (`scipy.sparse.csr_matrix`_):
            The matrix to force to be positive definite.
        min_eigenvalue (:obj:`float`, optional):
            The minimum allowed matrix eigenvalue.
        renormalize (:obj:`bool`, optional):
            Include the renormalization (last) step in the list above.

    Returns:
        `scipy.sparse.csr_matrix`_: The modified matrix.
    """
    if not isinstance(mat, sparse.csr_matrix):
        raise TypeError('Must provide a scipy.sparse.csr_matrix to impose_positive_definite.')
    # Get the eigenvalues/eigenvectors
    # WARNING: I didn't explore why to deeply, but scipy.sparse.linalg.eigs
    # provided *significantly* different results. They also seem to be worse in
    # the sense that the reconstructed matrix based on the adjusted eigenvalues
    # is more different than input matrix compared to the use of
    # numpy.linalg.eig.
    # NOTE: This command can take a while, depending on the size of the
    # array...
    w, v = map(lambda x : np.real(x), np.linalg.eig(mat.toarray()))
    if np.all(w > 0):
        # Already positive definite
        return mat
    # Force a minimum eigenvalue
    w = np.maximum(w, min_eigenvalue)
    # Reconstruct with the new eigenvalues
    _mat = np.dot(v, np.dot(np.diag(w), v.T))
    if not renormalize:
        return sparse.csr_matrix(_mat)
    # Renormalize
    d = mat.diagonal()
    t = 1./np.sqrt(np.diag(_mat))
    return sparse.csr_matrix(_mat * np.outer(t,t) * np.sqrt(np.outer(d,d)))


def is_positive_definite(mat, quiet=True):
    r"""
    Check if a matrix is positive definite.

    This is done by calculating the eigenvalues and eigenvectors of the
    provided matrix and checking if all the eigenvalues are :math:`>0`.
    Because of that, it is nearly as expensive as just calling
    :func:`impose_positive_definite`.

    Args:
        mat (`scipy.sparse.csr_matrix`_):
            The matrix to check.
        quiet (:obj:`bool`, optional):
            Suppress terminal output.

    Returns:
        :obj:`bool`: Flag that matrix is positive definite.
    """
    if not isinstance(mat, sparse.csr_matrix):
        raise TypeError('Must provide a scipy.sparse.csr_matrix to is_positive_definite.')
    # Get the eigenvalues/eigenvectors
    w, v = map(lambda x : np.real(x), np.linalg.eig(mat.toarray()))
    notpos = np.logical_not(w > 0)
    if not quiet:
        if np.any(notpos):
            warnings.warn('{0} eigenvalues are not positive!')
            print('{0:>6} {1:>8}'.format('Index', 'EigenVal'))
            for i in np.where(notpos)[0]:
                print('{0:>6} {1:8.2e}'.format(i, w[i]))
    return not np.any(notpos)


def cinv(mat, check_finite=False, upper=False):
    r"""
    Use Cholesky decomposition to invert a matrix.

    Args:
        mat (`numpy.ndarray`_, `scipy.sparse.csr_matrix`_):
            The array to invert.
        check_finite (:obj:`bool`, optional):
            Check that all the elements of ``mat`` are finite. See
            `scipy.linalg.cholesky`_ and `scipy.linalg.solve_triangular`.
        upper (:obj:`bool`, optional):
            Return only the upper triangle matrix that can be used to
            construct the inverse matrix. I.e., for input matrix
            :math:`\mathbf{M}`, this returns matrix :math:`\mathbf{U}` such
            that :math:`\mathbf{M}^{-1} = \mathbf{U} \mathbf{U}^T`.

    Returns:
        `numpy.ndarray`_: Inverse of the input matrix.
    """
    _mat = mat.toarray() if isinstance(mat, sparse.csr.csr_matrix) else mat
    # This uses scipy.linalg, not numpy.linalg
    cho = linalg.cholesky(_mat, check_finite=check_finite)
    # Returns an upper triangle matrix that can be used to construct the inverse matrix (see below)
    cho = linalg.solve_triangular(cho, np.identity(cho.shape[0]), check_finite=check_finite)
    # TODO: Make it a sparse matrix if upper?
    return cho if upper else np.dot(cho, cho.T)


def boxcar_replicate(arr, boxcar):
    """
    Boxcar replicate an array.

    Args:
        arr (`numpy.ndarray`_):
            Array to replicate.
        boxcar (:obj:`int`, :obj:`tuple`):
            Integer number of times to replicate each pixel. If a
            single integer, all axes are replicated the same number
            of times. If a :obj:`tuple`, the integer is defined
            separately for each array axis; length of tuple must
            match the number of array dimensions.

    Returns:
        `numpy.ndarray`_: The block-replicated array.
    """
    # Check and configure the input
    _boxcar = (boxcar,)*arr.ndim if isinstance(boxcar, int) else boxcar
    if not isinstance(_boxcar, tuple):
        raise TypeError('Input `boxcar` must be an integer or a tuple.')
    if len(_boxcar) != arr.ndim:
        raise ValueError('Must provide an integer or tuple with one number per array dimension.')

    # Perform the boxcar average over each axis and return the result
    _arr = arr.copy()
    for axis, box in zip(range(arr.ndim), _boxcar):
        _arr = np.repeat(_arr, box, axis=axis)
    return _arr


def inverse(array):
    """
    Calculate ``1/array``, enforcing positivity and setting values <= 0 to
    zero.
    
    The input array should be a quantity expected to always be positive, like
    a variance or an inverse variance. The quantity returned is::

        out = (array > 0.0)/(np.abs(array) + (array == 0.0))

    Args:
        array (`numpy.ndarray`_):
            Array to element-wise invert

    Returns:
        `numpy.ndarray`: The result of the element-wise inversion.
    """
    return (array > 0.0)/(np.abs(array) + (array == 0.0))


def sigma_clip_stdfunc_mad(data, **kwargs):
    """
    A simple wrapper for `scipy.stats.median_abs_deviation`_ that omits NaN
    values and rescales the output to match a normal distribution for use in
    `astropy.stats.sigma_clip`_.

    Args:
        data (`numpy.ndarray`_):
            Data to clip.
        **kwargs:
            Passed directly to `scipy.stats.median_abs_deviation`_.

    Returns:
        scalar-like, `numpy.ndarray`_: See `scipy.stats.median_abs_deviation`_.
    """
    return stats.median_abs_deviation(data, **kwargs, nan_policy='omit', scale='normal')


# TODO: Instead apply eps to the error (i.e., we don't want the weight to be
# large)?
def construct_ivar_weights(error, eps=None):
    r"""
    Produce inverse-variance weights based on the input errors.

    Weights are set to 0 if the error is :math:`<=0` or if the inverse
    variance is less than ``eps``.

    Args:
        error (`numpy.ndarray`_):
            Error to use to construct weights.
        eps (:obj:`float`, optional):
            The minimum allowed weight. Any weight (inverse variance) below
            this value is set to 0. If None, no minimum to the inverse
            variance is enforced.

    Returns:
        `numpy.ndarray`_: The inverse variance weights.
    """
    indx = error > 0
    wgts = np.zeros(error.shape, dtype=float)
    wgts[indx] = 1.0/error[indx]**2
    if eps is not None:
        wgts[wgts < eps] = 0.
    return wgts


def aggregate_stats(x, y, ye=None, wgts=None, gpm=None, eps=None, fill_value=None):
    """
    Construct a set of aggregate statistics for the provided data.

    Args:
        x (`numpy.ndarray`_):
            Independent coordinates
        y (`numpy.ndarray`_):
            Dependent coordinates
        ye (`numpy.ndarray`_, optional):
            Errors in the dependent coordinates. Used to construct inverse
            variance weights. If not provided, no inverse-variance weights
            are applied.
        wgts (`numpy.ndarray`_, optional):
            Weights to apply. Ignored if errors are provided. If None and no
            errors are provided (``ye``), uniform weights are applied.
        gpm (`numpy.ndarray`_, optional):
            Good-pixel mask used to select data to include. If None, all data
            are included.
        eps (:obj:`float`, optional):
            Minimum allowed weight. Any weight below this value is set to 0.
        fill_value (:obj:`float`, optional):
            If the statistics cannot be determined, replace the output with
            this fill value.

    Returns:
        :obj:`tuple`: The unweighted median y value, the unweighted median
        absolute deviation rescaled to match the standard deviation, the
        unweighted mean x, the unweighted mean y, the unweighted standard
        deviation of y, the error-weighted mean x, the error-weighted mean y,
        the error-weighted standard deviation of y, the error-weighted error
        in the mean y, the number of data points aggregated (any value with a
        non-zero weight), and a boolean `numpy.ndarray`_ with flagging the
        data included in the calculation.
    """
    # Weights
    _wgts = (np.ones(x.size, dtype=float) if wgts is None else wgts) \
                if ye is None else construct_ivar_weights(ye, eps=eps)
    indx = _wgts > 0
    if gpm is not None:
        indx &= gpm

    # Number of aggregated data points
    nbin = np.sum(indx)
    if nbin == 0:
        # Default values are all set to None
        return (fill_value,)*9 + (0, indx)

    # Unweighted statistics
    uwmed = np.median(y[indx])
    uwmad = sigma_clip_stdfunc_mad(y[indx])

    uwxbin = np.mean(x[indx])
    uwmean = np.mean(y[indx])
    uwsdev = np.sqrt(np.dot(y[indx]-uwmean,y[indx]-uwmean)/(nbin-1)) if nbin > 1 else fill_value

    # Weighted statistics
    # TODO: Include covariance
    wsum = np.sum(_wgts[indx])
    ewxbin = np.dot(_wgts[indx],x[indx])/wsum
    ewmean = np.dot(_wgts[indx],y[indx])/wsum
    ewsdev = np.dot(_wgts[indx],y[indx]**2)/wsum - ewmean**2
    ewsdev = fill_value if ewsdev < 0 or nbin <= 1 else np.sqrt(ewsdev*nbin/(nbin-1))
    ewerr = np.sqrt(1./wsum)

    return uwmed, uwmad, uwxbin, uwmean, uwsdev, ewxbin, ewmean, ewsdev, ewerr, nbin, indx


def _select_rej_stat(rej_stat, ewmean, ewsdev, uwmean, uwsdev, uwmed, uwmad):
    """
    Select and return the desired rejection statistic.
    """
    if rej_stat == 'ew':
        return ewmean, ewsdev
    if rej_stat == 'uw':
        return uwmean, uwsdev
    if rej_stat == 'ro':
        return uwmed, uwmad

    raise ValueError('rej_stat must be ew, uw, or ro.')


def clipped_aggregate_stats(x, y, ye=None, wgts=None, gpm=None, eps=None, fill_value=None,
                            sig_rej=None, rej_stat='ew', maxiter=None):
    """
    Construct a set of aggregate statistics for the provided data with
    iterative rejection.

    This method iteratively executes :func:`aggregate_stats` with rejection
    iterations. If ``sig_rej`` is None, this is identical to a single
    execution of :func:`aggregate_stats`.

    Args:
        x (`numpy.ndarray`_):
            Independent coordinates
        y (`numpy.ndarray`_):
            Dependent coordinates
        ye (`numpy.ndarray`_, optional):
            Errors in the dependent coordinates. Used to construct inverse
            variance weights. If not provided, no inverse-variance weights
            are applied.
        wgts (`numpy.ndarray`_, optional):
            Weights to apply. Ignored if errors are provided. If None and no
            errors are provided (``ye``), uniform weights are applied.
        gpm (`numpy.ndarray`_, optional):
            Good-pixel mask used to select data to include. If None, all data
            are included.
        eps (:obj:`float`, optional):
            Minimum allowed weight. Any weight below this value is set to 0.
        fill_value (:obj:`float`, optional):
            If the statistics cannot be determined, replace the output with
            this fill value.
        sig_rej (:obj:`float`, optional):
            The symmetric rejection threshold in units of the standard
            deviation.  If None, no rejection is performed.
        use_ew_stats (:obj:`str`, optional):
            The statistic to use when determining which values to reject.
            Allowed options are:

                - 'ew': Use the error-weighted mean and standard deviation
                - 'uw': Use the unweighted mean and standard deviation
                - 'ro': Use the robust statisitics, the unweighted median and
                   median absolute deviation (where the latter is normalized
                   to nominally match the standard deviation)
                
        maxiter (:obj:`int`, optional):
            Maximum number of rejection iterations; ``maxiter = 1`` means
            there are *no* rejection iterations. If None, iterations continue
            until no more data are rejected.
            
    Returns:
        :obj:`tuple`: The unweighted median y value, the unweighted median
        absolute deviation rescaled to match the standard deviation, the
        unweighted mean x, the unweighted mean y, the unweighted standard
        deviation of y, the error-weighted mean x, the error-weighted mean y,
        the error-weighted standard deviation of y, the error-weighted error
        in the mean y, and the number of data points aggregated (any value
        with a non-zero weight).
    """
    # Run the first iteration. The weights and good-pixel mask are defined here
    # so that they don't need to be redetermined for each call to
    # aggregate_stats
    _wgts = (np.ones(x.size, dtype=float) if wgts is None else wgts) \
                if ye is None else construct_ivar_weights(ye, eps=eps)
    _gpm = _wgts > 0
    if gpm is not None:
        _gpm &= gpm

    # Get the stats
    uwmed, uwmad, uwxbin, uwmean, uwsdev, ewxbin, ewmean, ewsdev, ewerr, nbin, new_gpm \
            = aggregate_stats(x, y, wgts=_wgts, gpm=_gpm, fill_value=fill_value)

    if nbin == 0 or sig_rej is None or maxiter == 1:
        # If there were no data includes or the rejection sigma is not
        # provided, then we're done
        return uwmed, uwmad, uwxbin, uwmean, uwsdev, ewxbin, ewmean, ewsdev, ewerr, nbin, new_gpm

    _gpm &= new_gpm
    i = 1
    while maxiter is None or i < maxiter:
        mean, sigma = _select_rej_stat(rej_stat, ewsdev, uwsdev, uwmad)
        rej = (y > mean + sig_rej*sigma) | (y < mean - sig_rej*sigma)
        if not np.any(rej):
            # Nothing was rejected so we're done
            return uwmed, uwmad, uwxbin, uwmean, uwsdev, ewxbin, ewmean, ewsdev, ewerr, nbin, _gpm
        # Include the rejection in the good-pixel mask
        _gpm &= np.logical_not(rej)
        uwmed, uwmad, uwxbin, uwmean, uwsdev, ewxbin, ewmean, ewsdev, ewerr, nbin, new_gpm \
                = aggregate_stats(x, y, wgts=_wgts, gpm=_gpm, fill_value=fill_value)
        _gpm &= new_gpm
        i += 1


def bin_stats(x, y, bin_center, bin_width, ye=None, wgts=None, gpm=None, eps=None, fill_value=None,
              sig_rej=None, rej_stat='ew', maxiter=None):
    r"""
    Compute aggregate statistics for a set of bins.

    This method runs :func:`clipped_aggregate_stats` on the data in each bin.
    The bin centers and widths must be pre-defined. Bins are allowed to
    overlap.

    Args:
        x (`numpy.ndarray`_):
            Independent coordinates
        y (`numpy.ndarray`_):
            Dependent coordinates
        bin_center (`numpy.ndarray`_):
            The set of independent coordinates for the center of each bin.
        bin_width (`numpy.ndarray`_):
            The width of each bin.
        ye (`numpy.ndarray`_, optional):
            Errors in the dependent coordinates. Used to construct inverse
            variance weights. If not provided, no inverse-variance weights
            are applied.
        wgts (`numpy.ndarray`_, optional):
            Weights to apply. Ignored if errors are provided. If None and no
            errors are provided (``ye``), uniform weights are applied.
        gpm (`numpy.ndarray`_, optional):
            Good-pixel mask used to select data to include. If None, all data
            are included.
        eps (:obj:`float`, optional):
            Minimum allowed weight. Any weight below this value is set to 0.
        fill_value (:obj:`float`, optional):
            If the statistics cannot be determined, replace the output with
            this fill value.
        sig_rej (:obj:`float`, optional):
            The symmetric rejection threshold in units of the standard
            deviation.  If None, no rejection is performed.
        use_ew_stats (:obj:`str`, optional):
            The statistic to use when determining which values to reject.
            Allowed options are:

                - 'ew': Use the error-weighted mean and standard deviation
                - 'uw': Use the unweighted mean and standard deviation
                - 'ro': Use the robust statisitics, the unweighted median and
                   median absolute deviation (where the latter is normalized
                   to nominally match the standard deviation)
                
        maxiter (:obj:`int`, optional):
            Maximum number of rejection iterations; ``maxiter = 1`` means
            there are *no* rejection iterations. If None, iterations continue
            until no more data are rejected.

    Returns:
        :obj:`tuple`: Thirteen `numpy.ndarray`_ objects are returned: The
        coordinate of the bin centers (this is just the input ``bin_centers``
        array), the unweighted median y value, the unweighted median absolute
        deviation rescaled to match the standard deviation, the unweighted
        mean x, the unweighted mean y, the unweighted standard deviation of
        y, the error-weighted mean x, the error-weighted mean y, the
        error-weighted standard deviation of y, the error-weighted error in
        the mean y, the total number of data points in the bin (this excludes
        any data that are masked on input either because ``ye`` or wgt`` is
        not larger than 0 or ``gpm`` is False), the number of data points
        used in the aggregated statistics, and a boolean array selecting data
        that were included in any bin. The shape of all arrays is the same as
        the input ``bin_centers``, except for the last array which is the
        same shape as the input ``x``.
    """
    # Setup the weights and good-pixel mask for all of the data here so that
    # they don't need to be redetermined for each call to aggregate_stats.
    _wgts = (np.ones(x.size, dtype=float) if wgts is None else wgts) \
                if ye is None else construct_ivar_weights(ye, eps=eps)
    _gpm = _wgts > 0
    if gpm is not None:
        _gpm &= gpm

    # Setup the output arrays
    nbins = bin_center.size
    uwxbin = np.zeros(nbins, dtype=float)
    uwmed = np.zeros(nbins, dtype=float)
    uwmad = np.zeros(nbins, dtype=float)
    uwmean = np.zeros(nbins, dtype=float)
    uwsdev = np.zeros(nbins, dtype=float)
    ewxbin = np.zeros(nbins, dtype=float)
    ewmean = np.zeros(nbins, dtype=float)
    ewsdev = np.zeros(nbins, dtype=float)
    ewerr = np.zeros(nbins, dtype=float)
    ntot = np.zeros(nbins, dtype=int)
    nbin = np.zeros(nbins, dtype=int)
    all_bin_gpm = _gpm.copy()

    for i in range(nbins):
        binlim = bin_center[i] + np.array([-1.,1.])*bin_width[i]/2.
        bin_gpm = _gpm & (x > binlim[0]) & (x < binlim[1])
        ntot[i] = np.sum(bin_gpm)
        if ntot[i] == 0:
            continue

        uwmed[i], uwmad[i], uwxbin[i], uwmean[i], uwsdev[i], ewxbin[i], ewmean[i], ewsdev[i], \
            ewerr[i], nbin[i], _bin_gpm \
                    = clipped_aggregate_stats(x[bin_gpm], y[bin_gpm], wgts=_wgts[bin_gpm],
                                              fill_value=fill_value, sig_rej=sig_rej,
                                              rej_stat=rej_stat, maxiter=maxiter)
        all_bin_gpm[bin_gpm] = _bin_gpm

    return bin_center, uwmed, uwmad, uwxbin, uwmean, uwsdev, ewxbin, ewmean, ewsdev, ewerr, \
             ntot, nbin, all_bin_gpm


def select_major_axis(r, th, r_range=None, wedge=30.):
    r"""
    Return a boolean array that selects data near the major axis.

    Args:
        r (`numpy.ndarray`_):
            In-plane disk radius relative to the center.
        th (`numpy.ndarray`_):
            In-plane disk azimuth in *radians* relative to the receding side
            of the major axis.
        r_range (:obj:`str`, array-like, optional):
            The lower and upper limit of the radial range over which to
            measure the median rotation velocity. If None, the radial range
            is from 1/5 to 2/3 of the radial range within the selected wedge
            around the major axis. If 'all', use all data, regardless of
            their radius.
        wedge (:obj:`float`, optional):
            The :math:`\pm` wedge in *degrees* around the major axis to
            select.

    Returns:
        `numpy.ndarray`_: A boolean array selecting the data within the
        desired range of the major axis.
    """
    # Select the spaxels within the wedge around the major axis
    _wedge = np.radians(wedge)
    gpm = (th < _wedge) | (th > 2*np.pi - _wedge) \
            | ((th > np.pi - _wedge) & (th < np.pi + _wedge))

    if r_range == 'all':
        # Do not select based on radius
        return gpm

    # Select the spaxels within a relevant radial range
    if r_range is None:
        maxr = np.amax(r[gpm])
        r_range = [maxr/5., 2*maxr/3.]
    gpm[r < r_range[0]] = False
    gpm[r > r_range[1]] = False
    return gpm


def growth_lim(a, lim, fac=1.0, midpoint=None, default=[0., 1.]):
    """
    Set the plots limits of an array based on two growth limits.

    Args:
        a (array-like):
            Array for which to determine limits.
        lim (:obj:`float`):
            Fraction of the total range of the array values to cover. Should
            be in the range [0, 1].
        fac (:obj:`float`, optional):
            Factor to contract/expand the range based on the growth limits.
            Default is no change.
        midpoint (:obj:`float`, optional):
            Force the midpoint of the range to be centered on this value. If
            None, set to the median of the data.
        default (:obj:`list`, optional):
            Default range to return if `a` has no data.

    Returns:
        :obj:`list`: Lower and upper limits for the range of a plot of the
        data in `a`.
    """
    # Get the values to plot
    _a = a.compressed() if isinstance(a, np.ma.MaskedArray) else np.asarray(a).ravel()
    if len(_a) == 0:
        # No data so return the default range
        return default

    # Sort the values
    srt = np.ma.argsort(_a)

    # Set the starting and ending values based on a fraction of the
    # growth
    _lim = 1.0 if lim > 1.0 else lim
    start = int(len(_a)*(1.0-_lim)/2)
    end = int(len(_a)*(_lim + (1.0-_lim)/2))
    if end == len(_a):
        end -= 1

    # Set the full range and increase it by the provided factor
    Da = (_a[srt[end]] - _a[srt[start]])*fac

    # Set the midpoint if not provided
    mid = (_a[srt[start]] + _a[srt[end]])/2 if midpoint is None else midpoint

    # Return the range for the plotted data
    return [ mid - Da/2, mid + Da/2 ]


def atleast_one_decade(lim):
    """
    """
    lglim = np.log10(lim)
    if int(lglim[1]) - int(np.ceil(lglim[0])) > 0:
        return (10**lglim).tolist()
    m = np.sum(lglim)/2
    ld = lglim[0] - np.floor(lglim[0])
    fd = np.ceil(lglim[1]) - lglim[1]
    w = lglim[1] - m
    dw = ld*1.01 if ld < fd else fd*1.01
    _lglim = np.array([m - w - dw, m + w + dw])

    # TODO: The next few lines are a hack to avoid making the upper limit to
    # large. E.g., when lim = [ 74 146], the output is [11 1020]. This pushes
    # the middle of the range to lower values.
    dl = np.diff(_lglim)[0]
    if dl > 1 and dl > 3*np.diff(lglim)[0]:
        return atleast_one_decade([lim[0]/3,lim[1]])

    return atleast_one_decade((10**_lglim).tolist())
    

def pixelated_gaussian(x, c=0.0, s=1.0, density=False):
    """
    Construct a Gaussian function integrated over the width of each pixel.

    Args:
        x (`numpy.ndarray`_):
            Coordinates for each pixel. The pixels should be regularly and
            linearly sampled, but this **is not checked.***
        c (:obj:`float`, optional):
            The center of the Gaussian profile.
        s (:obj:`float`, optional):
            The standard deviation of the Gaussian profile.
        density (:obj:`bool`, optional):
            Return the density profile, instead of the profile integrated
            over each pixel; i.e.::

                dx = np.mean(np.diff(x))
                assert np.array_equal(pixelated_gaussian(x, density=True),
                                      pixelated_gaussian(x)/dx)

            should return true.
    
    Returns:
        `numpy.ndarray`_: The vector with the Gaussian function integrated
        over the width of each pixel.
    """
    n = np.sqrt(2.)*s
    d = np.asarray(x)-c
    dx = np.mean(np.diff(x))
    g = (special.erf((d+dx/2.)/n) - special.erf((d-dx/2.)/n))/2.
    return g/dx if density else g


def find_largest_coherent_region(a):
    """
    Find the largest coherent region in a 2D array.

    This is basically a wrapper for `scipy.ndimage.label`_ that associates
    adjacent pixels (including diagonally) into groups. The largest group is
    determined and a boolean array is returned that selects those pixels
    associated with that group.

    Args:
        a (`numpy.ndarray`_):
            A 2D array passed directly to `scipy.ndimage.label`_. Pulled from
            that documentation: "Any non-zero values in input are counted as
            features and zero values are considered the background."
            Perferrably this is an integer array.

    Returns:
        `numpy.ndarray`_: Boolean array with the same shape as the input that
        selects pixels that are part of the largest coherent group.
    """
    labels, n = ndimage.label(a, structure=np.ones((3,3), dtype=int))
    if n == 1:
        return labels == 1

    # Only keep the largest coherent structure
    uniq_labels, npix = np.unique(labels, return_counts=True)
    indx = uniq_labels != 0
    return labels == uniq_labels[indx][np.argmax(npix[indx])]

def equal_shape(arr1, arr2, fill_value=0):
    '''
    Take two 2D arrays and pad them to make them the same shape

    Args:
        arr1, arr2 (`numpy.ndarray`_):
            2D arrays that will be padded to be the same shape
        fill_value (:obj:`float`, optional):
            Fill value for the padding

    Returns:
        Tuple of `numpy.ndarray`_s that are padded versions of the input arrays
    '''

    #check for non 2D arrays
    if arr1.ndim != 2 or arr2.ndim != 2:
        raise ValueError('Can only accept 2D arrays')

    #trivial case
    if arr1.shape == arr2.shape:
        return arr1, arr2
    
    #iterate through axes to pad each one appropriately
    for i in range(arr1.ndim):

        #figure out which array is smaller on this axis
        if arr1.shape[i] < arr2.shape[i]:
            smaller = arr1
            bigger = arr2
            order = 'fwd'
        elif arr1.shape[i] > arr2.shape[i]:
            smaller = arr2
            bigger = arr1
            order = 'rev'
        else:
            continue
        
        #add padding until appropriate size
        while smaller.shape[i] != bigger.shape[i]:
            fill = np.full((1,smaller.shape[1-i]), fill_value)
            if i: fill = fill.T

            #odd size difference
            if (bigger.shape[i] - smaller.shape[i])%2:
                smaller = np.concatenate([smaller, fill], axis=i) 

            #even size difference
            else:
                smaller = np.concatenate([fill, smaller, fill], axis=i)
        
        if order == 'fwd': arr1, arr2 = [smaller, bigger]
        elif order == 'rev': arr2, arr1 = [smaller, bigger]
            
    return arr1, arr2

def trim_shape(arr1, arr2, fill_value=0):
    '''
    Take one 2D array and make it the same shape as the other through trimming
    and padding

    Args:
        arr1 (`numpy.ndarray`_):
            2D array to be reshaped
        arr2 (`numpy.ndarray`_):
            2D array with target shape
        fill_value (:obj:`float`, optional):
            Fill value for the padding

    Returns:
        `numpy.ndarray`_: reshaped version of `arr1` with dimensions of `arr2`
    '''

    #check for non 2D arrays
    if arr1.ndim != 2 or arr2.ndim != 2:
        raise ValueError('Can only accept 2D arrays')

    #trivial case
    if arr1.shape == arr2.shape:
        return arr1
    
    #iterate through axes to figure out which need to be padded/trimmed
    for i in range(arr1.ndim):

        #if smaller, pad the array until appropriate size
        while arr1.shape[i] < arr2.shape[i]:
            fill = np.full((1, arr1.shape[1-i]), fill_value)
            if i: fill = fill.T

            #odd size difference
            if (arr2.shape[i] - arr1.shape[i])%2:
                arr1 = np.concatenate([arr1, fill], axis=i) 

            #even size difference
            else:
                arr1 = np.concatenate([fill, arr1, fill], axis=i)
                
        #if bigger, trim down the outside
        while arr1.shape[i] > arr2.shape[i]:

            #odd size difference
            if (arr1.shape[i] - arr2.shape[i])%2:
                arr1 = arr1.take(range(arr1.shape[i]-1),i)

            #even size difference
            else:
                arr1 = arr1.take(range(arr1.shape[i]-1),i)
        
    return arr1
