import math
import numpy as np
import matplotlib.pyplot as plt

## plot the density fingerprints
def plot_density(psi_dict,
                 range_t,
                 k_start,
                 k_end,
                 dpi,
                 title='Density fingerprints',
                 linewidth = 5,
                 legend_fontsize = 10,
                 legend_ncol = 1,
                 legend_loc = 'upper right'):
    plt.figure(dpi=dpi)
    fine_t = len(psi_dict[0])
    for k in range(k_start, k_end+1):
        plt.plot(np.linspace(range_t[0], range_t[1], fine_t),
                 psi_dict[k],
                 label = r'$\psi_{'+r'{}'.format(k)+r'}$',
                 linewidth = linewidth)
        plt.legend(fontsize = legend_fontsize,
                   ncol=legend_ncol,
                   loc=legend_loc)
        plt.title(title)
        plt.legend()
