import numpy as np
import matplotlib.pyplot as plt
from dataset_operations import get_umbrella_data, select_within_timecourse
import error_estimation as ee
from pyemma.thermo import estimate_umbrella_sampling
import pickle
'''
    Compute PMFs from umbrella sampling data.

    A typical workflow would be -
        Get data using get_umbrella_data
        Select your force constants (will need to convert to units of kT, and with angular coordinates there may need
                                     to be some radian/degree conversions)
        Figure out which combo of lags and maxiter lead to converged tram profiles using scan_params()
            - plot_series() is helpful for checking this output

        Run make_production_pmf to do the following (for angle systems)
            -make 1 pmf with all the good data
            -make a number of other pmfs with chunks of the data
            -align the pmfs onto the main pmf using a least squared regression
            -calculate error bars from the standard deviations as standard error of the mean (*1.96 for 95% confidence interval)
'''


def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def assignToCenters(time_series, centers):
    indices = np.zeros(time_series.size, dtype=int)
    for i, datapoint in enumerate(time_series):
        indices[i] = np.argmin(np.abs(datapoint - centers))
    return indices


def kJ_per_mol_to_J(fc):
    '''
        Convert a force constant from the Gromacs standard kJ / mol * distance^2 to (in terms of energy) unitless
        kT / distance ^2.

        (kj / mol) * (1000 J / 1 kJ) * (1 mol / Na molecules)
    '''
    return fc * 1000 / ( 6.022 * (10 ** 23))


def kT_in_J(T):
    ''' Calculates kt for a given temperature
        Yields Joules
    '''
    return T * 1.38064852 * 10 ** -23


def kt_in_kj_per_mol(T):
    ''' Calculates kt in kJ/mol '''
    return kT_in_J(T) * 6.022 * 10 ** 23 / 1000


def singleWhamTramProf(unfiltered_data, umbrella_centers, force_constants, bins, minframes=500):
    '''
        Given a series of umbrella timecourses, will extract the good data and run wham and tram on it.

        Note that if the Pymbar anaylsis shows that there are fewer than minframes good frames of data in a window,
        the last minframes frames from that window will be used, to ensure that we have umbrella coverage. This obviously
        should not be done for production PMFS (and I've seen a LOT of change to PMFs from dipping into this
        unequilibrated data)
    '''
    series_quality = ee.analyze_group_of_time_series(unfiltered_data, identifiers=umbrella_centers)
    frameranges = [ (np.min((first, unfiltered_data[i].size - minframes)), 10000) for i, first in enumerate(series_quality.t0)  ]
    filtered_data = select_within_timecourse(unfiltered_data, frameranges)
    us_dtrajs = [ assignToCenters(i, bins) for i in filtered_data  ]
    trammo = estimate_umbrella_sampling(filtered_data, us_dtrajs, umbrella_centers, force_constants, estimator='dtram' )
    whammo = estimate_umbrella_sampling(filtered_data, us_dtrajs, umbrella_centers, force_constants, estimator='wham' )
    return whammo, trammo


def multipleTramProfs(unfiltered_data, umbrella_centers, force_constants, bins, numProfs, minframes=500, maxiter=200000, maxerr=1.0E-10):
    ''' computes multiple tram profiles, dividing each window into numProfs segments of size n / N

        Returns a list of individual tram profiles. Beware - if you don't have sufficient data points, one or more of these
        profiles could be wonky, and the tram.f may have fewer data points than you expect
    '''
    series_quality = ee.analyze_group_of_time_series(unfiltered_data, identifiers=umbrella_centers)
    frameranges = [ (np.min((first, unfiltered_data[i].size - minframes)), 10000) for i, first in enumerate(series_quality.t0)  ]

    # get individual windows
    trams = []
    chunk_sizes = [ int((len(unfiltered_data[i]) - framerange[0])  / numProfs) for i, framerange in enumerate(frameranges) ]
    for i in range(numProfs):
        chunked_framerange = []
        for n, framerange in enumerate(frameranges):
            startframe = framerange[0] + i * chunk_sizes[n]
            endframe = startframe + chunk_sizes[n]
            chunked_framerange.append((startframe, endframe))
        filtered_data = select_within_timecourse(unfiltered_data, chunked_framerange)
        us_dtrajs = [ assignToCenters(i, bins) for i in filtered_data  ]
        trams.append(estimate_umbrella_sampling(filtered_data, us_dtrajs, umbrella_centers, force_constants, estimator='dtram', maxiter=maxiter, maxerr=maxerr ))
    return trams


def align_profs_deviation(profs, reference_prof):
    '''
        Translate a series of pmfs to best match a reference pmf, judged by a least-squared error score

        Paramters
            profs - list of tram objects (these are acted upon in place)
            reference_prof - single pmf (not a tram object, just np array)
    '''
    align_possibilities = np.linspace(-10, 10, 10000)
    for prof in profs:
        minval = 10 ** 6
        minindex = 0
        for ind, alignment in enumerate(align_possibilities):
            aligned_prof = prof.f - alignment
            val = np.sum((aligned_prof - reference_prof) ** 2)
            if val < minval:
                minindex = ind
                minval = val
        prof.f -= align_possibilities[minindex]


def plotProfs(replicates, average):
    for i, prof in enumerate(replicates):
        plt.plot(prof.f, label=str(i))
    plt.plot(average, linewidth=3, c='k', label="average")
    plt.legend()
    plt.show()


def standardError(replicates):
    '''
        Calculates standard error of the mean for a series of tram profiles. Assumes these are decorrelated in time

        Parameters
            replicates - list of tram objects
        Returns
            numpy array of standard errors for each data point
    '''
    datapoints = np.zeros((len(replicates), len(replicates[0].f)))
    for i, rep in enumerate(replicates):
        datapoints[i, :] = rep.f
    return np.std(datapoints, axis=0) / np.sqrt(datapoints.shape[0])


def param_scanning(unfiltered_data, force_constants, lags=(1, 2, 4, 8, 16, 32), iters=(5000, 10000, 20000, 40000, 80000, 200000)):
    '''
        Scans tram parameters lag and maxiter to check for convergence of PMFS. Visualize results with plotProfs
    '''
    series_quality = ee.analyze_group_of_time_series(unfiltered_data, identifiers=umbrella_centers)
    frameranges = [ (np.min((first, unfiltered_data[i].size - minframes)), 10000) for i, first in enumerate(series_quality.t0)  ]
    filtered_data = select_within_timecourse(unfiltered_data, frameranges)
    us_dtrajs = [ assignToCenters(i, bins) for i in filtered_data  ]
    tram_by_lags = estimate_umbrella_sampling(filtered_data, us_dtrajs, umbrella_centers, force_constants, estimator='dtram' , lag=lags, maxiter=200000, maxerr=1.0E-10)

    tram_by_iters = []
    for it in iters:
        tram_by_iters.append(estimate_umbrella_sampling(filtered_data, us_dtrajs, umbrella_centers, force_constants, estimator='dtram' , maxiter=it))
    return tram_by_lags, tram_by_iters


def make_production_angle_pmf(directory, force_constants, suffix="degrees.xvg", lag=30, maxerr=1.0E-10, maxiter=200000,
                              umbrella_centers=np.arange(0, 91, 3), bins=np.arange(0, 91)):
    '''
        Calculates a PMF and error for a data series specified in directory. This is meant for angular PMFs but could
        be adapted for distance PMFs without too much trouble
    '''
    umbrella_centers = [float(i) for i in umbrella_centers]
    unfiltered_data, dt = get_umbrella_data(directory, suffix, list(umbrella_centers), int)
    series_quality = ee.analyze_group_of_time_series(unfiltered_data, identifiers=umbrella_centers)
    frameranges = [ (first, 10000) for i, first in enumerate(series_quality.t0)  ]
    filtered_data = select_within_timecourse(unfiltered_data, frameranges)
    us_dtrajs = [ assignToCenters(i, bins) for i in filtered_data  ]
    trammo = estimate_umbrella_sampling(filtered_data, us_dtrajs, umbrella_centers, force_constants, estimator='dtram', maxerr=maxerr, maxiter=maxiter)
    trammo_chunked = multipleTramProfs(unfiltered_data, umbrella_centers, force_constants, bins, 4, maxerr=maxerr, maxiter=maxiter)
    align_profs_deviation(trammo_chunked, trammo.f)
    standardErrors = standardError(trammo_chunked)
    return trammo.f, standardErrors


def make_production_distance_pmf(directory, force_constants, suffix="nm.xvg", lag=30, maxerr=1.0E-10, maxiter=200000,
                                 umbrella_centers=np.arange(10, 40, 0.5), bins=np.arange(100, 40), reps=4):
    '''
        Calculates a PMF and error for a data series specified in directory. Meant for distance series
    '''
    umbrella_centers = [float(i) for i in umbrella_centers]
    unfiltered_data, dt = get_umbrella_data(directory, suffix, list(umbrella_centers), float)
    series_quality = ee.analyze_group_of_time_series(unfiltered_data, identifiers=umbrella_centers)
    frameranges = [ (first, 10000) for i, first in enumerate(series_quality.t0)  ]
    filtered_data = select_within_timecourse(unfiltered_data, frameranges)
    us_dtrajs = [ assignToCenters(i, bins) for i in filtered_data  ]
    trammo = estimate_umbrella_sampling(filtered_data, us_dtrajs, umbrella_centers, force_constants, estimator='dtram', maxerr=maxerr, maxiter=maxiter)
    trammo_chunked = multipleTramProfs(unfiltered_data, umbrella_centers, force_constants, bins, reps, maxerr=maxerr, maxiter=maxiter)
    align_profs_deviation(trammo_chunked, trammo.f)
    standardErrors = standardError(trammo_chunked)
    return trammo.f, standardErrors


def plot_series(tram_list, labels=None):
    if not labels:
        labels = np.arange(len(tram_list))
    for tram, label in zip(tram_list, labels):
        plt.plot(tram.f, label=label)
    plt.legend()
    plt.show()


if __name__ == "__main__":

    # -----------------------------------------------------------------------------------------------------------------------------
    # angle analysis - much more polished than the flat analysis ATM

    # -----------------------------------------------------------------------------------------------------------------------------

    regenerate_profiles = False
    if regenerate_profiles:
        angle_umbrella_windows = np.arange(0, 91, 3)
        angle_umbrella_bins  = np.arange(91)
        angle_force_constant = 12000   * np.pi * np.pi / (180 * 180)
        angle_force_constants = [ kJ_per_mol_to_J(angle_force_constant) / kT_in_J(310)] * angle_umbrella_windows.size
        minframes = 1000

        # paramaters chosen after look at scan_params
        maxiter = 100000
        maxerr = 1.0E-10
        angle_lag = 30

        rotate_path_5nm   = '/home/kevin/hdd/Projects/ATP/production/angles_longer/PC_5nm/production/px/pullx_'
        rotate_path_10nm   = '/home/kevin/hdd/Projects/ATP/production/angles_and_sizes/PC_10nm/production/px/pullx_'
        rotate_path_15nm   = '/home/kevin/hdd/Projects/ATP/production/angles_longer/PC_15nm/production/px/pullx_'

        f_5, se_5   = make_production_angle_pmf(rotate_path_5nm,  angle_force_constants, lag=angle_lag, maxiter=maxiter, maxerr=maxerr)
        f_10, se_10 = make_production_angle_pmf(rotate_path_10nm, angle_force_constants, lag=angle_lag, maxiter=maxiter, maxerr=maxerr)
        f_15, se_15 = make_production_angle_pmf(rotate_path_15nm, angle_force_constants, lag=angle_lag, maxiter=maxiter, maxerr=maxerr)

        pickle_save(f_5,  "polished_data/pmf_5nm.pkl")
        pickle_save(f_10, "polished_data/pmf_10nm.pkl")
        pickle_save(f_15, "polished_data/pmf_15nm.pkl")
        pickle_save(se_5,  "polished_data/se_5nm.pkl")
        pickle_save(se_10, "polished_data/se_10nm.pkl")
        pickle_save(se_15, "polished_data/se_15nm.pkl")

    f_5 = pickle_load("pmf_5nm.pkl")
    f_10 = pickle_load("pmf_10nm.pkl")
    f_15 = pickle_load("pmf_15nm.pkl")
    se_5  = pickle_load("se_15nm.pkl")
    se_10 = pickle_load("se_10nm.pkl")
    se_15 = pickle_load("se_15nm.pkl")

    # set overall minimum to 0, while aligning the profiles to be the same at 0 degrees
    f_5 -= np.min((f_5.min(), f_10.min(), f_15.min()))
    f_10 += (f_5[0] - f_10[0])
    f_15 += (f_5[0] - f_15[0])

    plt.figure()
    bins = np.arange(91)
    colors = ("r", "g", "b")
    for ind, (pmf, se, color) in enumerate(zip((f_5, f_10, f_15), (se_5, se_10, se_15), colors)):
        plt.plot(bins, pmf, label=str(5 * (ind + 1)), c=color)
        plt.fill_between(bins, pmf - 1.96 * se, pmf + 1.96 * se, alpha=0.5, edgecolor=color, facecolor=color)
    plt.legend()
    plt.show()

    # -----------------------------------------------------------------------------------------------------------------------------
    # distance analysis. Should work for both flat and cylindrical systems
    # -----------------------------------------------------------------------------------------------------------------------------

    ''' PC flat pulling '''
    PC_flat_path   = '/home/kevin/hdd/Projects/ATP/experimenting/flat_pull/production/px/pullx_'
    distances = np.arange(12.0, 40, 0.5)
    bins = np.arange(12, 40, 1)
    force_constant = 200   # kJ / mol nm^2
    umbrella_centers = [float(i) for i in distances]

    force_constants = [ kJ_per_mol_to_J(force_constant) / kT_in_J(310)] * distances.size
    f_dist_flat, se_dist_flat = make_production_distance_pmf(PC_flat_path, force_constants, bins=bins, umbrella_centers=distances, lag=5, reps=2)

    ''' PC cylinder pulling '''
    PC_10nm_distance_path   = '/home/kevin/hdd/Projects/ATP/production/distance_restraint/PC_10nm/production/px/pullx_'
    distances = np.arange(14.0, 28.0, 0.5)
    bins = np.arange(14.0, 28.0, 1)
    force_constant = 200   # kJ / mol nm^2
    minframes = 1000
    umbrella_centers = [float(i) for i in distances]
    force_constants = [ kJ_per_mol_to_J(force_constant) / kT_in_J(310)] * distances.size
    f_dist_cyl, se_dist_cyl = make_production_distance_pmf(PC_flat_path, force_constants, bins=bins, umbrella_centers=umbrella_centers, lag=5)
