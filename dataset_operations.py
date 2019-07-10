import numpy as np
'''
    Performs loading and filtering operations on data sets that consist of a list of np arrays

'''


def get_umbrella_data(prefix, suffix, data_points, dtype):
    '''
        Load a set of time series of gromacs xvg files in a folder

        Parameters
            prefix - path to folder + any file prefix (eg /home..../px/pullx_)
            suffix - end identifier (eg nm.xvg or degrees.xvg)
            data   - should be a numpy array or list, and correspond to the data series name
            dtype  - int or float, for the file matching approach

        Returns
            data   - list of numpy arrays
            dt     - time unit between points in series
    '''
    if dtype == float:
        data = [np.loadtxt("{:s}{:2.1f}{:s}".format(prefix, data_point, suffix), comments=("#", "@"))[:, 1]
                for data_point in data_points]
        dt   = np.loadtxt("{:s}{:2.1f}{:s}".format(prefix, data_points[0], suffix), comments=("#", "@"))
        dt = dt[1, 0] - dt[0, 0]
    elif dtype == int:
        data = [np.loadtxt("{:s}{:d}{:s}".format(prefix, data_point, suffix), comments=("#", "@"))[:, 1]
                for data_point in data_points]
        dt   = np.loadtxt("{:s}{:d}{:s}".format(prefix, data_points[0], suffix), comments=("#", "@"))
        dt = dt[1, 0] - dt[0, 0]
    return data, dt


def select_within_timecourse(data, selection):
    '''
        Passes through the arrays of a dataset and filters them internally. This is used to e.g. select the same time
        points in each data set to analyze

        Makes a deep copy so original data is retained

        Parameters:
            data      - list of np arrays
            selection - slicing to be done on the sets. Can be a tuple, or list of tuples
                            -tuple          - the same (startIndex, finishIndex) filter is applied to each set
                            -list of tuples - each set is curated individually, length(list) should = len(data)
    '''

    # input checking
    if isinstance(selection, list):
        if len(data) != len(selection):
            raise ValueError("selection size is not the same as the data size")
    elif isinstance(selection, tuple):
        if len(selection) != 2:
            raise ValueError("selection is not a size 2 tuple")
        selection = [selection] * len(data)

    out_data = []
    for series, sel in zip(data, selection):
        out_data.append(np.array(series[sel[0]:sel[1]]))
    return out_data


def select_between_timecourses(data, selection):
    '''
        Parameters

            data      - list of np arrays
            selection - list of indices of data to keep

        Returns
            list of lists of np arrays, len = len(selection)
    '''
    return [data[i] for i in selection]
