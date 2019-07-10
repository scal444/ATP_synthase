import matplotlib.pyplot as plt
import numpy as np

plt.rc('font',  size=8)
plt.rc('axes',  titlesize=12)
plt.rc('axes',  labelsize=12)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

trammo_5  = np.loadtxt("tram_5nm.txt")
trammo_10 = np.loadtxt("tram_10nm.txt")
trammo_15 = np.loadtxt("tram_15nm.txt")

# multiply SE by 1.96 for 95% confidence interval
trammo_5_se  = np.loadtxt("tram_5nm_error.txt")
trammo_10_se = np.loadtxt("tram_10nm_error.txt")
trammo_15_se = np.loadtxt("tram_15nm_error.txt")

# 5 nm system is truncated due to PBC issues.
bins = np.arange(0, 91)
bins_5nm = bins[-1 * trammo_5.size:]

# I'm not sure how to align these profiles to each other. This example zeros out the 90 degree window
trammo_5 -= trammo_5[-1]
trammo_10 -= trammo_10[-1]
trammo_15 -= trammo_15[-1]

plt.figure()

plt.plot(bins_5nm, trammo_5, c='b', label="5 nm")
plt.fill_between(bins_5nm,
                 trammo_5 - 1.96 * trammo_5_se,
                 trammo_5 + 1.96 * trammo_5_se,
                 alpha=0.5, edgecolor='b', facecolor='b')

plt.plot(bins,    trammo_10, c='g', label="10 nm")
plt.fill_between(bins,
                 trammo_10 - 1.96 * trammo_10_se,
                 trammo_10 + 1.96 * trammo_10_se,
                 alpha=0.5, edgecolor='g', facecolor='g')

plt.plot(bins,    trammo_15, c='r', label="15 nm")
plt.fill_between(bins,
                 trammo_15 - 1.96 * trammo_15_se,
                 trammo_15 + 1.96 * trammo_15_se,
                 alpha=0.5, edgecolor='r', facecolor='r')
plt.legend()
plt.xlabel("angles (deg)")
plt.ylabel("Free energy / kT")
