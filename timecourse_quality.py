import numpy as np
import error_estimation as ee
from dataset_operations import get_umbrella_data
'''
    The scripts/functions in this module analyze the set of windows for umbrella sampling on an individual basis,
    and displays info about the group in a digestable manner
'''


def get_decorr_frame(data, threshold, treat_none_as=10 ** 6):
    results = np.zeros(len(data), dtype=int)
    decorr_plot_data = []
    ba_plot_data = []
    for i, time_series in enumerate(data):
        decorr_plot_data.append(ee.check_decorrelation(time_series, corr_thresh=threshold, plot=False, retval="decorr_plot"))
        ba_plot_data.append(ee.check_decorrelation(    time_series, corr_thresh=threshold, plot=False, retval="ba_data"))
        decorr_frame = ee.check_decorrelation(         time_series, corr_thresh=threshold, plot=False, retval="decorr_frame")
        if decorr_frame is None:
            decorr_frame = treat_none_as
        results[i] = decorr_frame
    return results, decorr_plot_data, ba_plot_data


def check_convergence_process(path, data, suffix, dtype):
    series_list, dt = get_umbrella_data(path, suffix, data, dtype)
    series_quality = ee.analyze_group_of_time_series(series_list, identifiers=data)
    series_quality.plot_data_with_t0()
    series_quality.plot_only_equilibrated_data()
    # frameranges = [ (first, 10000) for first in series_quality.t0  ]
    # truncated_series = select_within_timecourse(series_list, frameranges)
    # decorr_frame, decorr_plot_data, ba_plot_data = get_decorr_frame(truncated_series, 0.05)

    # for i in decorr_plot_data:
    #        plt.plot(i)
    # plt.show()

    # for i in ba_plot_data:
    #     plt.plot(i)
    # plt.show()


if __name__ == "__main__":

    PC_15nm_angle_path   = '/home/kevin/hdd/Projects/ATP/production/angles_longer/PC_15nm/production/px/pullx_'
    PC_15nm_angle_data   = np.arange(0, 91, 3)
    PC_15nm_angle_suffix = "degrees.xvg"
    decorr_frame = check_convergence_process(PC_15nm_angle_path, PC_15nm_angle_data, PC_15nm_angle_suffix, int)

    PC_flat_path   = '/home/kevin/hdd/Projects/ATP/experimenting/flat_pull/production/px/pullx_'
    PC_flat_data   = np.arange(8.0, 40, 0.5)
    PC_flat_suffix = "nm.xvg"

    decorr_frame = check_convergence_process(PC_flat_path, PC_flat_data, PC_flat_suffix, float)
    PC_5nm_angle_path   = '/home/kevin/hdd/Projects/ATP/production/angles_longer/PC_5nm/production/px/pullx_'
    PC_5nm_angle_data   = np.arange(0, 91, 3)
    PC_5nm_angle_suffix = "degrees.xvg"
    decorr_frame = check_convergence_process(PC_5nm_angle_path, PC_5nm_angle_data, PC_5nm_angle_suffix, int)

    PC_10nm_angle_path   = '/home/kevin/hdd/Projects/ATP/production/angles_and_sizes/PC_10nm/production/px/pullx_'
    PC_10nm_angle_data   = np.arange(0, 91, 3)
    PC_10nm_angle_suffix = "degrees.xvg"
    decorr_frame = check_convergence_process(PC_10nm_angle_path, PC_10nm_angle_data, PC_10nm_angle_suffix, int)

    PC_10nm_distance_path   = '/home/kevin/hdd/Projects/ATP/production/distance_restraint/PC_10nm/production/px/pullx_'
    PC_10nm_distance_data   = np.arange(10.0, 28, 0.5)
    PC_10nm_distance_suffix = "nm.xvg"
    decorr_frame = check_convergence_process(PC_10nm_distance_path, PC_10nm_distance_data, PC_10nm_distance_suffix, float)
