import numpy as np
from statsmodels.tsa.stattools import acf
from pymbar.timeseries import detectEquilibration
import matplotlib.pyplot as plt

'''
    Utilities for automated checking of data sets for error convergence.

    Block averaging is used to estimate error of a time course. At a long enough block length, blocks become
    decorrelated and can be used to estimate standard error. However, with insufficient sampling there may be too few
    blocks that are sufficiently independent to confidently estimate the error (as well as the actual value). Grossfield
    and co. suggest 20 independent "samples" is sufficient, I'd say 10 is still reasonable - it's not too far into the
    noise that you can't be confident your error has converged.
'''


class Data_quality:

    def __init__(self, data, identifiers=None):
        self.data = data
        self.identifiers = identifiers
        self.block_average_profiles = []
        self.n_samples_effective = []
        self.fract_of_series_used = []
        self.t0 = []

    def plot_data_with_t0(self):
        plt.figure()

        for i, series in enumerate(self.data):
            series_copy = np.array(series)
            series_copy -= series.mean()
            series_copy = 0.4 * series_copy /  np.max(np.abs(series_copy))
            series_copy += i
            t0 = self.t0[i]
            plt.plot(np.arange(t0), series_copy[:t0] , c='r')
            plt.plot(np.arange(t0, series_copy.size), series_copy[t0:], c='b')
        plt.xlabel("time points")
        plt.ylabel("normalized series")
        plt.show()

    def plot_only_equilibrated_data(self):
        plt.figure()

        for i, series in enumerate(self.data):
            series_copy = np.array(series)
            series_copy -= series.mean()
            series_copy = 0.4 * series_copy /  np.max(np.abs(series_copy))
            series_copy += i
            plt.plot(series_copy[self.t0[i]:], c='b')
        plt.xlabel("time points")
        plt.ylabel("normalized series")
        plt.show()

    def plot_t0(self):
        plt.figure()
        plt.plot(self.t0)
        plt.show()

    def plot_neff(self):
        plt.figure()
        plt.plot(self.n_samples_effective)
        plt.show()

    def plot_block_averages(self, firstframe=0):

        plt.figure()
        for i in self.block_average_profiles:
            plt.plot(i[firstframe:])
        plt.show()

    def get_reasonable_first_frame(self, plot=False, cutoff=0.75):
        cutoff_mask = np.array(self.fract_of_series_used) > cutoff
        return np.array(self.t0)[cutoff_mask].mean()

    def plot_average_BA(self):
        max = 0
        for i in self.block_average_profiles:
            max = np.max((len(i), max))
        average = np.zeros(max)
        counts = np.zeros(max)
        for i in self.block_average_profiles:
            average[:len(i)] += np.array(i)
            counts[:len(i)] += 1
        average /= counts
        plt.plot(average)


def block_average(data, blocksize, partial_block_cutoff_size=0.5):
    '''
        Calculates the block average error estimate for a 1D data set given a set block size.

        Parameters -
            -data                 - 1D numpy array
            -blocksize            - integer, number of data points in a block
            -partial_block_cutoff - float between 0 and 1 (inclusive). Used to determine whether to include the final
                                    block, which is typically smaller than the others. If the fractional size of the
                                    last block with respect to the requested blocksize >= partial_block_cutoff, the
                                    block is included. Set to 0 to always discard partial blocks, set to 0 to always
                                    include

        Returns
            -block standard error (BSE) - Standard error of sample from block size. BSE = std(block means)/sqrt(nblocks)
    '''
    # determine if last block should be included
    data_size = data.size
    last_block_size = (data_size % blocksize)
    last_block_fraction_size =  last_block_size / data.size
    if last_block_size  < partial_block_cutoff_size and last_block_fraction_size > 0:  # mod  = 0 if perfect division
        data_size -= last_block_size  # if cutting off last block, just reduce arange size

    data_range = np.arange(0, data_size, blocksize)

    M = data_range.size    # number of blocks. This is determined AFTER deciding to keep last block or not
    means = np.zeros(M)
    for index, start in enumerate(data_range):
        means[index] = data[start:start + blocksize].mean()
    return np.std(means) / np.sqrt(M)


def block_average_range(data, block_range, partial_block_cutoff_size=0.5):
    '''
        Calculates standard errors from block averaging over a range of block sizes

        Parameters
            -data        - 1D numpy array of time-course data
            -block_range - 1D numpy array of block sizes to estimate SE for - must start at 1 or greater
            -partial_block_cutoff_size - treatment of last block - see description in block_average function

        Returns
            -block standard errors - 1D numpy array, size of block_range. SEs for each block size
    '''
    bse = np.zeros(block_range.size)
    for index, blocksize in enumerate(block_range):
        bse[index] = block_average(data, blocksize, partial_block_cutoff_size=partial_block_cutoff_size)
    return bse


def check_decorrelation(data, min_samples=10, corr_thresh=0, plot=True, retval="ba_data"):
    '''
        Takes a data set, figures out the maximum allowable blocksize (totalsize / 10, or whatever other criteria the
        user specifies), then see if the sample actually decorrelates in that time. The basic indicator of convergence
        is whether or not the block averaging profile flattens out in this regime, but it can be unclear on two fronts.
            First, with not enough data, one can convince themself that the curve has flattened within noise when really
        they have only 2 or 3 points in the data set at that block. Limiting the block average curve to block sizes that
        provide at least 10 blocks should stop that issue, and let you know if you are actually converged with that
        amount of data. So - if your curve is still rising at the end of this plot, you likely do not have enough data.
            Second, when one has data that is actually decorrelated from frame to frame (or within a few frames), the
        block average curve will not show it's typical asymptote. This should be a good thing, you've got great
        sampling! But without a clear curve how can you tell it's a decorrelated sample and not some analysis error?
        One check is to run an acf alongside the block average. If the acf shows a rapid (within a few frame taus)
        decrease to 0, you're golden.

        Parameters:
            data - 1D numpy array
            min_samples - data must have at LEAST this many samples in block averaging, sets upper bound on block size
                          as data.size / min_samples
            corr_thresh - for an acf, used to report when the acf falls BELOW this level for the first time. Indicates
                          that the sample is decorrelated at this time lag
    '''

    n_obs = data.size

    # round will help to clean up end of dataset issues. If halfway or more to another block, will add that block.
    # Otherwise will keep blocksize lower
    max_block_size = int(np.round(n_obs / min_samples))

    ba_data = block_average_range(data, np.arange(1, max_block_size))
    acf_data = acf(data, nlags=max_block_size)

    is_decorrelated = np.any(acf_data <= corr_thresh)

    if is_decorrelated:
        decorr_frame = np.argmax(acf_data < corr_thresh)
    else:
        decorr_frame = None
    if plot:
        f, axarr = plt.subplots(2, sharex=True)

        axarr[0].set_ylabel('BSE')
        axarr[0].plot(np.arange(1, ba_data.size + 1), ba_data)
        if is_decorrelated:
            axarr[0].plot([decorr_frame, decorr_frame], [ba_data.min(), ba_data.max()], 'k--')

        axarr[1].set_ylabel('ACF')
        axarr[1].set_xlabel('Block size (top) / tau (bot)')
        axarr[1].plot(np.arange(acf_data.size), acf_data)   # nlags + 1, with lag=0
        axarr[1].plot([0, max_block_size], [corr_thresh, corr_thresh])
        plt.show()

    if retval == "ba_data":
        return ba_data
    elif retval == "decorr_frame":
        return decorr_frame
    elif retval == "decorr_plot":
        return acf_data


def assess_equilibration(timeseries, minimum_fraction_of_series=0.2, minimum_effective_samples=10,
                         crash_on_bad_series=False, plot=False):
    '''
        Determines whether a time series samples equilibrium, and whether the equilibrated portion of the time series
        contains sufficient decorrelated samples. Wraps pymbar.timeseries.detectEquilibration

        Parameters:
            timeseries                 - 1D numpy array of data
            minimum_fraction_of_series - the pymbar utility will determine the amount of data to toss from the beginning
                                         of the simulation. If not equilibrated, it will only take the last few data
                                         points, which is an artefact. If the selected data is below this fraction of
                                         the total data, the function will report that we are not equilibrated
            minimum_effective_samples  - the pymbar utility will report the number of decorrelated samples. Warn if
                                         below this number
            crash_on_bad_series        - if data series does not pass inspection, throw an error if true. Else just
                                         warns by printing
            plot                       - plots the equilibrium and nonequilibrium regions of the series if true
        Returns
            t0                         - initial frame to analyze reported from pymbar
            num_effective_samples      - number of effective samples reported by pymbar

            // TODO check that input is 1D
    '''

    t0, g, neff = detectEquilibration(timeseries)

    is_good_series = True
    fraction_of_sample_retained = (timeseries.size - t0) / timeseries.size
    if fraction_of_sample_retained < minimum_fraction_of_series:
        is_good_series = False
        print("Warning - only {:2.1f}% of the data set is selected by pymbar".format(100 * fraction_of_sample_retained) +
              "- usually a sign that you do not have an equilibrated sample")
    # do elif to avoid extra messages
    elif neff < minimum_effective_samples:
        is_good_series = False
        print("This data series has only {:d} effective samples, less than the {:d} sample minimum".format(int(neff), minimum_effective_samples))

    if plot:
        plt.figure()
        plt.xlabel("series data point")
        plt.ylabel("series value")
        plt.plot(np.arange(t0),                  timeseries[:t0], "r", label="unequilibrated data")
        plt.plot(np.arange(t0, timeseries.size), timeseries[t0:], "b", label="equilibrated data")
        plt.show()

    if crash_on_bad_series and not is_good_series:
        raise Exception("Bad data series")

    return t0, neff


def analyze_group_of_time_series(timeseries_list, identifiers=None):
    '''
        Compile a bunch of analyses of a list of time series, from e.g. a set of PMF windows

        Returns a dataQuality instance
    '''
    data_struct = Data_quality(timeseries_list, identifiers=identifiers)
    data_struct.first_frames = []
    data_struct.block_average_profiles = []
    data_struct.n_samples_effective = []
    data_struct.fract_of_series_used = []
    for i in range(len(timeseries_list)):
        data_struct.block_average_profiles.append(check_decorrelation(timeseries_list[i], plot=False))
        t0, neff = assess_equilibration(timeseries_list[i])
        series_len = len(timeseries_list[i])
        fract_of_data_used = (series_len - t0) / series_len
        data_struct.fract_of_series_used.append(fract_of_data_used)
        data_struct.t0.append(t0)
        data_struct.n_samples_effective.append(neff)
    return data_struct
