
import warnings

from IPython import embed

import numpy as np
from scipy import sparse, linalg

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
    to calculate the binned values::
    
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
        1D vector with the list of unique bin IDs. Shape is :math:`(N_{\rm
        bin},)`. If ``binid`` is not provided, this is returned as None.

    ubin_indx : `numpy.ndarray`_
        The index vector used to select the unique bin values from a
        flattened map of binned data, *excluding* any any element with
        ``binid == -1``. Shape is :math:`(N_{\rm bin},)`. If ``binid`` is not
        provided, this is identical to ``grid_indx``. These indices can be
        used to reconstruct the list of unique bins; i.e.::

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

    grid_indx = np.arange(np.prod(_spatial_shape), dtype=int)
    if binid is None:
        # All bins are valid and considered unique
        bin_transform = sparse.coo_matrix((np.ones(np.prod(spatial_shape), dtype=float),
                                           (grid_indx,grid_indx)),
                                          shape=(np.prod(spatial_shape),)*2).tocsr()
        return None, grid_indx.copy(), grid_indx, grid_indx.copy(), bin_transform

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

    return ubinid, ubin_indx, grid_indx, bin_inverse, bin_transform


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
        _arr = numpy.repeat(_arr, box, axis=axis)
    return _arr


def cinv(mat, check_finite=False, upper=False):
    r"""
    Use Cholesky decomposition to invert a matrix.

    Args:
        mat (`numpy.ndarray`_):
            The dense numpy array to invert.
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
    # This uses scipy.linalg, not numpy.linalg
    cho = linalg.cholesky(mat, check_finite=check_finite)
    # Returns an upper triangle matrix that can be used to construct the inverse matrix (see below)
    cho = linalg.solve_triangular(cho, np.identity(cho.shape[0]), check_finite=check_finite)
    return cho if upper else np.dot(cho, cho.T)


