import pickle
import os
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from KB_python.coordinate_manipulation.transformations import cart2pol


def pickle_save(data, filename):
    '''
        Wrapper around pickle to serialize data

        Parameters
            data - data to serialize
            filename - where to serialize data
    '''
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle)


def pickle_load(filename):
    '''
        Wrapper around pickle to deserialize data

        Parameters:
            filename - pickle file to load
        Returns:
            loaded data
    '''
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
        return data


def separate_leaflets(radii):
    '''
        Calculates inner and outer leaflets of a cylinder based on radius

        Parameters:
            Radii - n_sims * n_frames * n_particles
        Returns:
            inner, outer leaflet indices
    '''
    r_mean = radii.mean()
    particle_means = radii.mean(axis=0).mean(axis=0)
    return np.where(particle_means < r_mean)[0], np.where(particle_means >= r_mean)[0]


def get_cylindrical_coordinates(trajectories, lipid_indices, prot_indices, n_frames):
    n_trajectories = len(trajectories)
    n_lipids = lipid_indices.size
    lipid_theta = np.zeros((n_trajectories, n_frames, n_lipids))
    lipid_rho   =  np.zeros((n_trajectories, n_frames, n_lipids))
    lipid_z     = np.zeros((n_trajectories, n_frames, n_lipids))
    prot_z = np.zeros(n_trajectories)

    for i, traj in enumerate(trajectories):

        firstframe = traj.n_frames - n_frames

        prot_z[i] = traj.xyz[firstframe:, prot_indices, 2].mean()

        # can't use center_coordinates to center as protein would throw off the COM
        # center xy based on lipids, z based on protein
        traj.xyz[firstframe:, :, :2] -= traj.xyz[firstframe:, lipid_indices, :2].mean(axis=1)[:, np.newaxis, :]
        traj.xyz[firstframe:, :, 2] -= prot_z[i]

        theta, rho, z = cart2pol(traj.xyz[firstframe:, lipid_indices, :])

        lipid_theta[i, :, :] = theta
        lipid_rho[  i, :, :] = rho
        lipid_z[    i, :, :] = z

    return lipid_theta * 180 / np.pi, lipid_rho, lipid_z


def get_heatmaps(theta, rho, z, theta_bins, z_bins):
    return [binned_statistic_2d(theta[i, :, :].flatten(), z[i, :, :].flatten(), rho[i, :, :].flatten(),
                                statistic='mean', bins=[theta_bins, z_bins])[0] for i in range(theta.shape[0])]


def plot_heatMap(data, zdims=(-35, 35), thetadims=(-180, 180), minc=-6, maxc=6, title=None):
    plt.imshow(data.T, cmap="seismic",
               interpolation="spline16",
               vmin=minc, vmax=maxc,
               extent=(*thetadims, *zdims))
    cbar = plt.colorbar()
    cbar.set_label(label="difference from mean (nm)")
    plt.xlabel("angle (degrees)")
    plt.ylabel("z (nm)")
    if title:
        plt.title(title)

    plt.show()


def set_nans_to_value(heatmaps, value):
    for heatmap in heatmaps:
        heatmap[np.isnan(heatmap)] = value

if __name__ == "__main__":

    topdir = '/home/kevin/hdd/Projects/ATP/production/angles_longer/PC_15nm/production/trajectories'
    pdb_path = os.path.join(topdir, "firstframe.pdb")
    trajectories = [md.load_xtc(os.path.join(topdir, "{}degrees.xtc".format(i)), top=pdb_path) for i in range(0, 91, 3)]

    lipid_indices = trajectories[0].topology.select("resname POPC")
    prot_indices  = trajectories[0].topology.select("not resname POPC")
    theta, rho, z = get_cylindrical_coordinates(trajectories, lipid_indices, prot_indices, 50)

    theta_bins = np.arange(-180, 180 )
    z_bins = np.arange(-35, 35, 0.5)

    heatmaps = get_heatmaps(theta, rho, z, theta_bins, z_bins)

    rho_mean = rho.mean()
    set_nans_to_value(heatmaps, rho_mean)

    for i, heatmap in enumerate(heatmaps):
        plot_heatMap(heatmap - rho_mean, title="{} degrees".format(i * 3))
