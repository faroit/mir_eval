# -*- coding: utf-8 -*-
'''
Source separation algorithms attempt to extract recordings of individual
sources from a recording of a mixture of sources.  Evaluation methods for
source separation compare the extracted sources from reference sources and
attempt to measure the perceptual quality of the separation.

Currently, only bss_eval is implemented, as described in:
    Emmanuel Vincent, Rémi Gribonval, and Cédric Févotte, "Performance
    measurement in blind audio source separation," IEEE Trans. on Audio,
    Speech and Language Processing, 14(4):1462-1469, 2006.

See also the bss_eval MATLAB toolbox:
    http://bass-db.gforge.inria.fr/bss_eval/

Conventions
-----------

An audio signal is expected to be in the format of a 1-dimensional array where
the entries are the samples of the audio signal.  When providing a group of
estimated or reference sources, they should be provided in a 2-dimensional
array, where the first dimension corresponds to the source number and the
second corresponds to the samples.

Metrics
-------

* :func:`mir_eval.separation.bss_eval_sources`: Computes the bss_eval metrics,
  which optimally match the estimated sources to the reference sources and
  measure the distortion and artifacts present in the estimated sources as well
  as the interference between them.

'''

import numpy as np
import scipy.fftpack
from scipy.linalg import toeplitz
from scipy.signal import fftconvolve
import collections
import itertools
import warnings
from . import util


# The maximum allowable number of sources (prevents insane computational load)
MAX_SOURCES = 100


class DefaultFFT(object):
    """Class designed for power-users to inherit and replace with their own
    versions of fft, ifft and convolution. This allows substitution of faster
    versions that may require more installation and setup than is appropriate
    for mir_eval.

    Defaults to using scipy.fftpack and scipy.signal.fftconvolve

    Attributes:
        fft_name : string
            the name of the currently planned fft
        input_array : ndarray
            the array on which to perform the fft
        direction : string
            the direction of the fft that will be performed next either
            'forward' or 'inverse'
        source_type : dtype
            the type of the input to the fft
        result_type : dtype
            the desired output type
    """

    def __init__(self):
        """Returns a new object."""
        self.fft_name = ""
        self.input_array = None
        self.direction = ""
        self.source_type = None
        self.result_type = None

    def plan_fft(self, fft_name="", input_array=None, direction='forward',
                 source_type=None, result_type=None):
        """Plan the next fft to be performed. Simply `pass` if no planning
        required. A few parameters are supplied that should be sufficient for
        most planning algorithms. More may have to be added depending on the
        planning used."""
        self.fft_name = fft_name
        self.input_array = input_array
        self.direction = direction
        self.source_type = source_type
        self.result_type = result_type

    def perform_fft(self, target, **kwargs):
        """Perform an fft (that was just planned, if planning is required)."""
        return scipy.fftpack.fft(target, **kwargs)

    def perform_ifft(self, target, **kwargs):
        """Perform an ifft (that was just planned, if planning is required)."""
        return scipy.fftpack.ifft(target, **kwargs)

    def perform_convolution(self, tensor1, tensor2, **kwargs):
        """Perform a convolution of the two input tensors. May need to make use
        of above functions for performing the convolution."""
        return fftconvolve(tensor1, tensor2, **kwargs)


def validate(reference_sources, estimated_sources):
    """Checks that the input data to a metric are valid, and throws helpful
    errors if not.

    Parameters
    ----------
    reference_sources : np.ndarray, shape=(nsrc1, nsampl1)
        matrix containing true sources
    estimated_sources : np.ndarray, shape=(nsrc2, nsampl2)
        matrix containing estimated sources

    """

    if reference_sources.shape != estimated_sources.shape:
        raise ValueError('The shape of estimated sources and the true '
                         'sources should match.  reference_sources.shape '
                         '= {}, estimated_sources '
                         '= {}'.format(reference_sources.shape,
                                       estimated_sources.shape))

    if reference_sources.size == 0:
        warnings.warn("reference_sources is empty, should be of size "
                      "(nsrc, nsample).  sdr, sir, sar, and perm will all "
                      "be empty np.ndarrays")
    elif np.any(np.all(reference_sources == 0, axis=1)):
        raise ValueError('All the reference sources should be non-silent (not '
                         'all-zeros), but at least one of the reference '
                         'sources is all 0s, which introduces ambiguity to the'
                         ' evaluation. (Otherwise we can add infinitely many '
                         'all-zero sources.)')

    if estimated_sources.size == 0:
        warnings.warn("estimated_sources is empty, should be of size "
                      "(nsrc, nsample).  sdr, sir, sar, and perm will all "
                      "be empty np.ndarrays")
    elif np.any(np.all(estimated_sources == 0, axis=1)):
        raise ValueError('All the estimated sources should be non-silent (not '
                         'all-zeros), but at least one of the estimated '
                         'sources is all 0s. Since we require each reference '
                         'source to be non-silent, having a silent estiamted '
                         'source will result in an underdetermined system.')

    if estimated_sources.shape[0] > MAX_SOURCES:
        raise ValueError('The supplied matrices should be of shape (n_sources,'
                         ' n_samples) but estimated_sources.shape[0] = {} '
                         'which is greater than '
                         'mir_eval.separation.MAX_SOURCES = {}.  To override '
                         'this check, set mir_eval.separation.MAX_SOURCES to '
                         'a larger value.'.format(estimated_sources.shape[0],
                                                  MAX_SOURCES))

    if reference_sources.shape[0] > MAX_SOURCES:
        raise ValueError('The supplied matrices should be of shape (n_sources,'
                         ' n_samples) but reference_sources.shape[0] = {} '
                         'which is greater than '
                         'mir_eval.separation.MAX_SOURCES = {}.  To override '
                         'this check, set mir_eval.separation.MAX_SOURCES to '
                         'a larger value.'.format(estimated_sources.shape[0],
                                                  MAX_SOURCES))


def bss_eval_sources(reference_sources, estimated_sources,
                     compute_permutation=True, fft_evaluator=None):
    """MATLAB translation of BSS_EVAL Toolbox

    Ordering and measurement of the separation quality for estimated source
    signals in terms of filtered true source, interference and artifacts.

    The decomposition allows a time-invariant filter distortion of length
    512, as described in Section III.B of [#vincent2006performance]_.

    Examples
    --------
    >>> # reference_sources[n] should be an ndarray of samples of the
    >>> # n'th reference source
    >>> # estimated_sources[n] should be the same for the n'th estimated
    >>> # source
    >>> (sdr, sir, sar,
    ...  perm) = mir_eval.separation.bss_eval_sources(reference_sources,
    ...                                               estimated_sources)

    Parameters
    ----------
    reference_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing true sources (must have same shape as
        estimated_sources)
    estimated_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing estimated sources (must have same shape as
        reference_sources)
    compute_permutation : bool, optional
        compute permutation of estimate/source combinations (True by default)
    fft_evaluator : object, optional
        instance of class with methods for evaluating ffts

    Returns
    -------
    sdr : np.ndarray, shape=(nsrc,)
        vector of Signal to Distortion Ratios (SDR)
    sir : np.ndarray, shape=(nsrc,)
        vector of Source to Interference Ratios (SIR)
    sar : np.ndarray, shape=(nsrc,)
        vector of Sources to Artifacts Ratios (SAR)
    perm : np.ndarray, shape=(nsrc,)
        vector containing the best ordering of estimated sources in
        the mean SIR sense (estimated source number perm[j] corresponds to
        true source number j)
        Note: perm will be [0, 1, ..., nsrc-1] if compute_permutation is False

    """

    # make sure the input is of shape (nsrc, nsampl)
    if estimated_sources.ndim == 1:
        estimated_sources = estimated_sources[np.newaxis, :]
    if reference_sources.ndim == 1:
        reference_sources = reference_sources[np.newaxis, :]

    validate(reference_sources, estimated_sources)
    # If empty matrices were supplied, return empty lists (special case)
    if reference_sources.size == 0 or estimated_sources.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    nsrc = estimated_sources.shape[0]

    # check if use provided their own fft evaluator
    if fft_evaluator is None:
        fft_evaluator = DefaultFFT()
    else:
        # verify that all the functions can be called
        assert callable(fft_evaluator.plan_fft)
        assert callable(fft_evaluator.perform_fft)
        assert callable(fft_evaluator.perform_ifft)
        assert callable(fft_evaluator.perform_convolution)

    # does user desire permutations?
    if compute_permutation:
        # compute criteria for all possible pair matches
        sdr = np.empty((nsrc, nsrc))
        sir = np.empty((nsrc, nsrc))
        sar = np.empty((nsrc, nsrc))
        for jest in range(nsrc):
            for jtrue in range(nsrc):
                s_true, e_spat, e_interf, e_artif = \
                    _bss_decomp_mtifilt(reference_sources,
                                        estimated_sources[jest],
                                        jtrue, 512, fft_evaluator)
                sdr[jest, jtrue], sir[jest, jtrue], sar[jest, jtrue] = \
                    _bss_source_crit(s_true, e_spat, e_interf, e_artif)
    else:
        # compute criteria for only the simple correspondence
        # (estimate 1 is estimate corresponding to reference source 1, etc.)
        sdr = np.empty(nsrc)
        sir = np.empty(nsrc)
        sar = np.empty(nsrc)
        for j in range(nsrc):
            s_true, e_spat, e_interf, e_artif = \
                _bss_decomp_mtifilt(reference_sources,
                                    estimated_sources[j],
                                    j, 512, fft_evaluator)
            sdr[j], sir[j], sar[j] = \
                _bss_source_crit(s_true, e_spat, e_interf, e_artif)

    # does user desire permutations?
    if compute_permutation:
        # select the best ordering
        perms = list(itertools.permutations(list(range(nsrc))))
        mean_sir = np.empty(len(perms))
        dum = np.arange(nsrc)
        for (i, perm) in enumerate(perms):
            mean_sir[i] = np.mean(sir[perm, dum])
        popt = perms[np.argmax(mean_sir)]
        idx = (popt, dum)
        return (sdr[idx], sir[idx], sar[idx], np.asarray(popt))
    else:
        # return the default permutation for compatibility
        popt = np.asarray(list(range(nsrc)))
        return (sdr, sir, sar, popt)


def bss_eval_sources_framewise(reference_sources, estimated_sources,
                               window, hop, compute_permutation=False,
                               fft_evaluator=None):
    """Framewise computation of bss_eval_sources

    Examples
    --------
    >>> # reference_sources[n] should be an ndarray of samples of the
    >>> # n'th reference source
    >>> # estimated_sources[n] should be the same for the n'th estimated
    >>> # source
    >>> (sdr, sir, sar,
    ...  perm) = mir_eval.separation.bss_eval_sources_framewise(
             reference_sources,
    ...      estimated_sources)

    Parameters
    ----------
    reference_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing true sources (must have the same shape as
        estimated_sources)
    estimated_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing estimated sources (must have the same shape as
        reference_sources)
    window : int
        Window length for framewise evaluation
    hop : int
        Hop size for framewise evaluation
    compute_permutation : bool, optional
        compute permutation of estimate/source combinations for all windows
        (False by default)

    Returns
    -------
    sdr : np.ndarray, shape=(nsrc, nframes)
        vector of Signal to Distortion Ratios (SDR)
    sir : np.ndarray, shape=(nsrc, nframes)
        vector of Source to Interference Ratios (SIR)
    sar : np.ndarray, shape=(nsrc, nframes)
        vector of Sources to Artifacts Ratios (SAR)
    perm : np.ndarray, shape=(nsrc, nframes)
        vector containing the best ordering of estimated sources in
        the mean SIR sense (estimated source number perm[j] corresponds to
        true source number j)
        Note: perm will be range(nsrc) for all windows if compute_permutation
        is False

    """

    # make sure the input is of shape (nsrc, nsampl)
    if estimated_sources.ndim == 1:
        estimated_sources = estimated_sources[np.newaxis, :]
    if reference_sources.ndim == 1:
        reference_sources = reference_sources[np.newaxis, :]

    validate(reference_sources, estimated_sources)
    # If empty matrices were supplied, return empty lists (special case)
    if reference_sources.size == 0 or estimated_sources.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    nsrc = reference_sources.shape[0]

    nwin = int(
        np.floor((reference_sources.shape[1] - window + hop) / hop)
    )
    # make sure that more than 1 window will be evaluated
    if nwin < 2:
        raise ValueError('Invalid window size and hop size have been supplied.'
                         'From these paramters it was determined that only {} '
                         'window(s) should be used.'.format(nwin))

    # check if use provided their own fft evaluator
    if fft_evaluator is None:
        fft_evaluator = DefaultFFT()
    else:
        # verify that all the functions can be called
        assert callable(fft_evaluator.plan_fft)
        assert callable(fft_evaluator.perform_fft)
        assert callable(fft_evaluator.perform_ifft)
        assert callable(fft_evaluator.perform_convolution)

    # compute the criteria across all windows
    sdr = np.empty((nsrc, nwin))
    sir = np.empty((nsrc, nwin))
    sar = np.empty((nsrc, nwin))
    perm = np.empty((nsrc, nwin))

    # k iterates across all the windows
    for k in range(nwin):
        win_slice = slice(k * hop, k * hop + window)
        sdr[:, k], sir[:, k], sar[:, k], perm[:, k] = bss_eval_sources(
            reference_sources[:, win_slice],
            estimated_sources[:, win_slice],
            compute_permutation,
            fft_evaluator
        )

    return sdr, sir, sar, perm


def bss_eval_images(reference_sources, estimated_sources,
                    compute_permutation=True, fft_evaluator=None):
    """Implementation of the bss_eval_images function from the
    BSS_EVAL Matlab toolbox.

    Ordering and measurement of the separation quality for estimated source
    signals in terms of filtered true source, interference and artifacts.
    This method also provides the ISR measure.

    The decomposition allows a time-invariant filter distortion of length
    512, as described in Section III.B of [#vincent2006performance]_.

    Passing True for compute_permutation will much improve the performance of
    the evaluation; however, it is not always appropriate and is not the
    way that the BSS_EVAL Matlab toolbox computes bss_eval_images.

    Examples
    --------
    >>> # reference_sources[n] should be an ndarray of samples of the
    >>> # n'th reference source
    >>> # estimated_sources[n] should be the same for the n'th estimated
    >>> # source
    >>> (sdr, isr, sir, sar,
    ...  perm) = mir_eval.separation.bss_eval_images(reference_sources,
    ...                                               estimated_sources)

    Parameters
    ----------
    reference_sources : np.ndarray, shape=(nsrc, nsampl, nchan)
        matrix containing true sources
    estimated_sources : np.ndarray, shape=(nsrc, nsampl, nchan)
        matrix containing estimated sources
    compute_permutation : bool, optional
        compute permutation of estimate/source combinations (True by default)

    Returns
    -------
    sdr : np.ndarray, shape=(nsrc,)
        vector of Signal to Distortion Ratios (SDR)
    isr : np.ndarray, shape=(nsrc,)
        vector of source Image to Spatial distortion Ratios (ISR)
    sir : np.ndarray, shape=(nsrc,)
        vector of Source to Interference Ratios (SIR)
    sar : np.ndarray, shape=(nsrc,)
        vector of Sources to Artifacts Ratios (SAR)
    perm : np.ndarray, shape=(nsrc,)
        vector containing the best ordering of estimated sources in
        the mean SIR sense (estimated source number perm[j] corresponds to
        true source number j)
        Note: perm will be (1,2,...,nsrc) if compute_permutation is False

    """

    # make sure the input is of shape (nsrc, nsampl, nchan)
    if estimated_sources.ndim != 3 or reference_sources.ndim != 3:
        raise ValueError('Please ensure input is in the form (nsrc, nsampl,'
                         'nchan). Cannot infer desired shape from input.')

    validate(reference_sources, estimated_sources)
    # If empty matrices were supplied, return empty lists (special case)
    if reference_sources.size == 0 or estimated_sources.size == 0:
        return np.array([]), np.array([]), np.array([]), \
                         np.array([]), np.array([])

    # check if use provided their own fft evaluator
    if fft_evaluator is None:
        fft_evaluator = DefaultFFT()
    else:
        # verify that all the functions can be called
        assert callable(fft_evaluator.plan_fft)
        assert callable(fft_evaluator.perform_fft)
        assert callable(fft_evaluator.perform_ifft)
        assert callable(fft_evaluator.perform_convolution)

    # determine size parameters
    nsrc = estimated_sources.shape[0]
    nsampl = estimated_sources.shape[1]
    nchan = estimated_sources.shape[2]

    # does the user desire permutation?
    if compute_permutation:
        # compute criteria for all possible pair matches
        sdr = np.empty((nsrc, nsrc))
        isr = np.empty((nsrc, nsrc))
        sir = np.empty((nsrc, nsrc))
        sar = np.empty((nsrc, nsrc))
        for jest in range(nsrc):
            for jtrue in range(nsrc):
                s_true, e_spat, e_interf, e_artif = \
                    _bss_decomp_mtifilt_images(
                        reference_sources,
                        np.reshape(
                            estimated_sources[jest],
                            (nsampl, nchan),
                            order='F'
                        ),
                        jtrue,
                        512,
                        fft_evaluator
                    )
                sdr[jest, jtrue], isr[jest, jtrue], \
                    sir[jest, jtrue], sar[jest, jtrue] = \
                    _bss_image_crit(s_true, e_spat, e_interf, e_artif)
    else:
        # compute criteria for only the simple correspondence
        # (estimate 1 is estimate corresponding to reference source 1, etc.)
        sdr = np.empty(nsrc)
        isr = np.empty(nsrc)
        sir = np.empty(nsrc)
        sar = np.empty(nsrc)
        Gj = [0] * nsrc  # prepare G matrics with zeroes
        G = np.zeros(1)
        for j in range(nsrc):
            # save G matrix to avoid recomputing it every call
            s_true, e_spat, e_interf, e_artif, Gj_temp, G = \
                _bss_decomp_mtifilt_images(reference_sources,
                                           np.reshape(estimated_sources[j],
                                                      (nsampl, nchan),
                                                      order='F'),
                                           j, 512, Gj[j], G, fft_evaluator)
            Gj[j] = Gj_temp
            sdr[j], isr[j], sir[j], sar[j] = \
                _bss_image_crit(s_true, e_spat, e_interf, e_artif)

    # does user desire permutations?
    if compute_permutation:
        # select the best ordering
        perms = list(itertools.permutations(range(nsrc)))
        mean_sir = np.empty(len(perms))
        dum = np.arange(nsrc)
        for (i, perm) in enumerate(perms):
            mean_sir[i] = np.mean(sir[perm, dum])
        popt = perms[np.argmax(mean_sir)]
        idx = (popt, dum)
        return (sdr[idx], isr[idx], sir[idx], sar[idx], popt)
    else:
        # return the default permutation for compatibility
        popt = list(range(nsrc))
        return (sdr, isr, sir, sar, popt)


def bss_eval_images_framewise(reference_sources, estimated_sources,
                              window, hop, compute_permutation=False,
                              fft_evaluator=None):
    """Framewise computation of bss_eval_images

    Examples
    --------
    >>> # reference_sources[n] should be an ndarray of samples of the
    >>> # n'th reference source
    >>> # estimated_sources[n] should be the same for the n'th estimated
    >>> # source
    >>> (sdr, sir, sar,
    ...  perm) = mir_eval.separation.bss_eval_images_framewise(
             reference_sources,
    ...      estimated_sources,
             window,
    ....     hop)

    Parameters
    ----------
    reference_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing true sources (must have the same shape as
        estimated_sources)
    estimated_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing estimated sources (must have the same shape as
        reference_sources)
    window : int
        Window length for framewise evaluation
    hop : int
        Hop size for framewise evaluation
    compute_permutation : bool, optional
        compute permutation of estimate/source combinations for all windows
        (False by default)

    Returns
    -------
    sdr : np.ndarray, shape=(nsrc, nframes)
        vector of Signal to Distortion Ratios (SDR)
    isr : np.ndarray, shape=(nsrc, nframes)
        vector of source Image to Spatial distortion Ratios (ISR)
    sir : np.ndarray, shape=(nsrc, nframes)
        vector of Source to Interference Ratios (SIR)
    sar : np.ndarray, shape=(nsrc, nframes)
        vector of Sources to Artifacts Ratios (SAR)
    perm : np.ndarray, shape=(nsrc, nframes)
        vector containing the best ordering of estimated sources in
        the mean SIR sense (estimated source number perm[j] corresponds to
        true source number j)
        Note: perm will be range(nsrc) for all windows if compute_permutation
        is False

    """

    # make sure the input is of shape (nsrc, nsampl)
    if estimated_sources.ndim == 1:
        estimated_sources = estimated_sources[np.newaxis, :]
    if reference_sources.ndim == 1:
        reference_sources = reference_sources[np.newaxis, :]

    validate(reference_sources, estimated_sources)
    # If empty matrices were supplied, return empty lists (special case)
    if reference_sources.size == 0 or estimated_sources.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    nsrc = reference_sources.shape[0]

    nwin = int(
        np.floor((reference_sources.shape[1] - window + hop) / hop)
    )
    # make sure that more than 1 window will be evaluated
    if nwin < 2:
        raise ValueError('Invalid window size and hop size have been supplied.'
                         'From these paramters it was determined that only {} '
                         'window(s) should be used.'.format(nwin))

    # check if use provided their own fft evaluator
    if fft_evaluator is None:
        fft_evaluator = DefaultFFT()
    else:
        # verify that all the functions can be called
        assert callable(fft_evaluator.plan_fft)
        assert callable(fft_evaluator.perform_fft)
        assert callable(fft_evaluator.perform_ifft)
        assert callable(fft_evaluator.perform_convolution)

    # compute the criteria across all windows
    sdr = np.empty((nsrc, nwin))
    isr = np.empty((nsrc, nwin))
    sir = np.empty((nsrc, nwin))
    sar = np.empty((nsrc, nwin))
    perm = np.empty((nsrc, nwin))

    # k iterates across all the windows
    for k in range(nwin):
        win_slice = slice(k * hop, k * hop + window)
        sdr[:, k], isr[:, k], sir[:, k], sar[:, k], perm[:, k] = \
            bss_eval_images(
                reference_sources[:, win_slice, :],
                estimated_sources[:, win_slice, :],
                compute_permutation,
                fft_evaluator
            )

    return sdr, isr, sir, sar, perm


def _bss_decomp_mtifilt(reference_sources, estimated_source, j, flen,
                        fft_evaluator):
    """Decomposition of an estimated source image into four components
    representing respectively the true source image, spatial (or filtering)
    distortion, interference and artifacts, derived from the true source
    images using multichannel time-invariant filters.

    Parameters
    ----------
    reference_sources :

    estimated_source :

    j :

    flen :

    fft_evaluator :


    Returns
    -------

    """
    nsampl = estimated_source.size
    # decomposition
    # true source image
    s_true = np.hstack((reference_sources[j], np.zeros(flen - 1)))
    # spatial (or filtering) distortion
    e_spat = _project(reference_sources[j, np.newaxis, :], estimated_source,
                      flen, fft_evaluator) - s_true
    # interference
    e_interf = _project(reference_sources, estimated_source,
                        flen, fft_evaluator) - s_true - e_spat
    # artifacts
    e_artif = -s_true - e_spat - e_interf
    e_artif[:nsampl] += estimated_source
    return (s_true, e_spat, e_interf, e_artif)


def _bss_decomp_mtifilt_images(reference_sources, estimated_source, j, flen,
                               fft_evaluator, Gj=None, G=None):
    """Decomposition of an estimated source image into four components
    representing respectively the true source image, spatial (or filtering)
    distortion, interference and artifacts, derived from the true source
    images using multichannel time-invariant filters.
    Adapted version to work with multichannel sources.
    Improved performance can be gained by passing Gj and G parameters.
    """
    nsampl = np.shape(estimated_source)[0]
    nchan = np.shape(estimated_source)[1]
    # are we saving the Gj and G parameters?
    saveg = Gj is not None and G is not None
    # decomposition
    # true source image
    s_true = np.hstack((np.reshape(reference_sources[j],
                                   (nsampl, nchan),
                                   order="F").transpose(),
                        np.zeros((nchan, flen - 1))))
    # spatial (or filtering) distortion
    if saveg:
        e_spat, Gj = _project_images(reference_sources[j, np.newaxis, :],
                                     estimated_source, flen, fft_evaluator, Gj)
    else:
        e_spat = _project_images(reference_sources[j, np.newaxis, :],
                                 estimated_source, flen, fft_evaluator)
    e_spat = e_spat - s_true
    # interference
    if saveg:
        e_interf, G = _project_images(reference_sources,
                                      estimated_source, flen, fft_evaluator, G)
    else:
        e_interf = _project_images(reference_sources,
                                   estimated_source, flen, fft_evaluator)
    e_interf = e_interf - s_true - e_spat
    # artifacts
    e_artif = -s_true - e_spat - e_interf
    e_artif[:, :nsampl] += estimated_source.transpose()
    # return Gj and G only if they were passed in
    if saveg:
        return (s_true, e_spat, e_interf, e_artif, Gj, G)
    else:
        return (s_true, e_spat, e_interf, e_artif)


def _project(reference_sources, estimated_source, flen, fft_evaluator):
    """Least-squares projection of estimated source on the subspace spanned by
    delayed versions of reference sources, with delays between 0 and flen-1

    Parameters
    ----------
    reference_sources :

    estimated_source :

    flen :

    fft_evaluator :


    Returns
    -------

    """
    nsrc, nsampl = reference_sources.shape

    # computing coefficients of least squares problem via FFT ##
    # zero padding and FFT of input data
    reference_sources = np.hstack((reference_sources,
                                   np.zeros((nsrc, flen - 1))))
    estimated_source = np.hstack((estimated_source, np.zeros(flen - 1)))
    n_fft = int(2**np.ceil(np.log2(nsampl + flen - 1.)))
    # perform an fft of the reference sources
    fft_evaluator.plan_fft('fft_sf', reference_sources, 'forward')
    sf = fft_evaluator.perform_fft(reference_sources, n=n_fft)
    # perform an fft of the estimated sources
    fft_evaluator.plan_fft('fft_sef', estimated_source, 'forward')
    sef = fft_evaluator.perform_fft(estimated_source, n=n_fft)
    # inner products between delayed versions of reference_sources
    G = np.zeros((nsrc * flen, nsrc * flen))
    for i in range(nsrc):
        for j in range(nsrc):
            ssf = sf[i] * np.conj(sf[j])
            fft_evaluator.plan_fft('fft_ssf', ssf, 'inverse')
            ssf = np.real(fft_evaluator.perform_ifft(ssf))
            ss = toeplitz(np.hstack((ssf[0], ssf[-1:-flen:-1])),
                          r=ssf[:flen])
            G[i * flen: (i+1) * flen, j * flen: (j+1) * flen] = ss
            G[j * flen: (j+1) * flen, i * flen: (i+1) * flen] = ss.T
    # inner products between estimated_source and delayed versions of
    # reference_sources
    D = np.zeros(nsrc * flen)
    for i in range(nsrc):
        ssef = sf[i] * np.conj(sef)
        fft_evaluator.plan_fft('fft_ssef', ssef, 'inverse')
        ssef = np.real(fft_evaluator.perform_ifft(ssef))
        D[i * flen: (i+1) * flen] = np.hstack((ssef[0], ssef[-1:-flen:-1]))

    # Computing projection
    # Distortion filters
    try:
        C = np.linalg.solve(G, D).reshape(flen, nsrc, order='F')
    except np.linalg.linalg.LinAlgError:
        C = np.linalg.lstsq(G, D)[0].reshape(flen, nsrc, order='F')
    # Filtering
    sproj = np.zeros(nsampl + flen - 1)
    for i in range(nsrc):
        sproj += fft_evaluator.perform_convolution(
            C[:, i],
            reference_sources[i]
        )[:nsampl + flen - 1]
    return sproj


def _project_images(reference_sources, estimated_source, flen, fft_evaluator,
                    G=None):
    """Least-squares projection of estimated source on the subspace spanned by
    delayed versions of reference sources, with delays between 0 and flen-1
    """
    nsrc, nsampl, nchan = reference_sources.shape
    reference_sources = np.reshape(np.transpose(reference_sources, (2, 0, 1)),
                                   (nchan*nsrc, nsampl), order='F')

    # computing coefficients of least squares problem via FFT ##
    # zero padding and FFT of input data
    reference_sources = np.hstack((reference_sources,
                                   np.zeros((nchan*nsrc, flen - 1))))
    estimated_source = \
        np.hstack((estimated_source.transpose(), np.zeros((nchan, flen - 1))))
    n_fft = int(2**np.ceil(np.log2(nsampl + flen - 1.)))
    # perform an fft of the reference sources
    fft_evaluator.plan_fft('fft_sf', reference_sources, 'forward')
    sf = fft_evaluator.perform_fft(reference_sources, n=n_fft)
    # perform an fft of the estimated sources
    fft_evaluator.plan_fft('fft_sef', estimated_source, 'forward')
    sef = fft_evaluator.perform_fft(estimated_source, n=n_fft)
    # inner products between delayed versions of reference_sources
    if G is None:
        saveg = False
        G = np.zeros((nchan * nsrc * flen, nchan * nsrc * flen))
        for i in range(nchan * nsrc):
            for j in range(i+1):
                ssf = sf[i] * np.conj(sf[j])
                fft_evaluator.plan_fft('fft_ssf', ssf, 'inverse')
                ssf = np.real(fft_evaluator.perform_ifft(ssf))
                ss = toeplitz(np.hstack((ssf[0], ssf[-1:-flen:-1])),
                              r=ssf[:flen])
                G[i * flen: (i+1) * flen, j * flen: (j+1) * flen] = ss
                G[j * flen: (j+1) * flen, i * flen: (i+1) * flen] = ss.T
    else:  # avoid recomputing G (only works if no permutation is desired)
        saveg = True  # return G
        if np.all(G == 0):  # only compute G if passed as 0
            G = np.zeros((nchan * nsrc * flen, nchan * nsrc * flen))
            for i in range(nchan * nsrc):
                for j in range(i+1):
                    ssf = sf[i] * np.conj(sf[j])
                    fft_evaluator.plan_fft('fft_ssf', ssf, 'inverse')
                    ssf = np.real(fft_evaluator.perform_ifft(ssf))
                    ss = toeplitz(np.hstack((ssf[0], ssf[-1:-flen:-1])),
                                  r=ssf[:flen])
                    G[i * flen: (i+1) * flen, j * flen: (j+1) * flen] = ss
                    G[j * flen: (j+1) * flen, i * flen: (i+1) * flen] = ss.T

    # inner products between estimated_source and delayed versions of
    # reference_sources
    D = np.zeros((nchan * nsrc * flen, nchan))
    for k in range(nchan * nsrc):
        for i in range(nchan):
            ssef = sf[k] * np.conj(sef[i])
            fft_evaluator.plan_fft('fft_ssef', ssef, 'inverse')
            ssef = np.real(fft_evaluator.perform_ifft(ssef))
            D[k * flen: (k+1) * flen, i] = \
                np.hstack((ssef[0], ssef[-1:-flen:-1])).transpose()

    # Computing projection
    # Distortion filters
    try:
        C = np.linalg.solve(G, D).reshape(flen, nchan*nsrc, nchan, order='F')
    except np.linalg.linalg.LinAlgError:
        C = np.linalg.lstsq(G, D)[0].reshape(flen, nchan*nsrc, nchan,
                                             order='F')
    # Filtering
    sproj = np.zeros((nchan, nsampl + flen - 1))
    for k in range(nchan * nsrc):
        for i in range(nchan):
            sproj += fft_evaluator.perform_convolution(
                C[:, k, i].transpose(),
                reference_sources[k]
            )[:nsampl + flen - 1]
    # return G only if it was passed in
    if saveg:
        return sproj, G
    else:
        return sproj


def _bss_source_crit(s_true, e_spat, e_interf, e_artif):
    """Measurement of the separation quality for a given source in terms of
    filtered true source, interference and artifacts.

    Parameters
    ----------
    s_true :

    e_spat :

    e_interf :

    e_artif :


    Returns
    -------

    """
    # energy ratios
    s_filt = s_true + e_spat
    sdr = _safe_db(np.sum(s_filt**2), np.sum((e_interf + e_artif)**2))
    sir = _safe_db(np.sum(s_filt**2), np.sum(e_interf**2))
    sar = _safe_db(np.sum((s_filt + e_interf)**2), np.sum(e_artif**2))
    return (sdr, sir, sar)


def _bss_image_crit(s_true, e_spat, e_interf, e_artif):
    """Measurement of the separation quality for a given image in terms of
    filtered true source, spatial error, interference and artifacts.

    Parameters
    ----------
    s_true :

    e_spat :

    e_interf :

    e_artif :


    Returns
    -------

    """
    # energy ratios
    sdr = _safe_db(np.sum(s_true**2), np.sum((e_spat+e_interf+e_artif)**2))
    isr = _safe_db(np.sum(s_true**2), np.sum(e_spat**2))
    sir = _safe_db(np.sum((s_true+e_spat)**2), np.sum(e_interf**2))
    sar = _safe_db(np.sum((s_true+e_spat+e_interf)**2), np.sum(e_artif**2))
    return (sdr, isr, sir, sar)


def _safe_db(num, den):
    """Properly handle the potential +Inf db SIR, instead of raising a
    RuntimeWarning. Only denominator is checked because the numerator can never
    be 0.

    Parameters
    ----------
    num :

    den :


    Returns
    -------

    """
    if den == 0:
        return np.Inf
    return 10 * np.log10(num / den)


def evaluate(reference_sources, estimated_sources, image=False,
             window=None, hop=None, **kwargs):
    """Compute all metrics for the given reference and estimated annotations.

    Examples
    --------
    >>> # reference_sources[n] should be an ndarray of samples of the
    >>> # n'th reference source
    >>> # estimated_sources[n] should be the same for the n'th estimated
    >>> scores = mir_eval.separation.evaluate(reference_sources,
    ...                                       estimated_sources)

    Parameters
    ----------
    reference_sources : np.ndarray, shape=(nsrc, nsampl[, nchan])
        matrix containing true sources
    estimated_sources : np.ndarray, shape=(nsrc, nsampl[, nchan])
        matrix containing estimated sources
    image : bool, optional
        whether to use bss_eval_images for evaluating the metrics
        if False, input cannot have more than 2 dimensions
    window : int, optional
        Window length for framewise evaluation
    hop : int, optional
        Hop size for framewise evaluation
    kwargs
        Additional keyword arguments which will be passed to the
        appropriate metric or preprocessing functions.

    Returns
    -------
    scores : dict
        Dictionary of scores, where the key is the metric name (str) and
        the value is the (float) score achieved.

    """
    # Parameter image must be True if input is higher than 3D
    if reference_sources.ndim > 2 or estimated_sources.ndim > 2:
        if not image:
            raise ValueError('Cannot evaluate a stereo image using'
                             'bss_eval_sources. Please pass image=True to'
                             'evaluate using bss_eval_images.')
    # Ensure image is a boolean (to prevent accidental parameter misordering)
    if not isinstance(image, bool):
        raise ValueError('The image parameter supplied is not a boolean!')

    # Compute all the metrics
    scores = collections.OrderedDict()

    if window is not None and hop is not None:
        if image:
            raise ValueError('Windowed evaluation is not yet implemented for'
                             'images.')
        else:
            sdr, sir, sar, perm = util.filter_kwargs(
                bss_eval_sources_framewise,
                reference_sources,
                estimated_sources,
                window,
                hop,
                **kwargs
            )
    elif window is not None or hop is not None:
        raise ValueError('In order to perform windowed evaluation, both window'
                         'and hop parameters must be supplied.')
    else:
        if image:
            sdr, sir, sar, perm = util.filter_kwargs(
                bss_eval_images,
                reference_sources,
                estimated_sources,
                **kwargs
            )
        else:
            sdr, sir, sar, perm = util.filter_kwargs(
                bss_eval_sources,
                reference_sources,
                estimated_sources,
                **kwargs
            )

    scores['Source to Distortion'] = sdr.tolist()
    scores['Source to Interference'] = sir.tolist()
    scores['Source to Artifact'] = sar.tolist()
    scores['Source permutation'] = perm

    return scores
