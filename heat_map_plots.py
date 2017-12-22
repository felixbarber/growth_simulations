import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
import matplotlib.cm as cmx
from shapely.geometry import LineString
from descartes import PolygonPatch
sns.set(font_scale=2)

def heat_map_1(obs, y, x, ax, xlabel=None, ylabel=None, title=None, bound=0.1, val=1.0, color='black', outline=True,
               alpha=1.0, cmap_lims=None):
    # Note that this assumes that obs[i,j] is produced with y[i], x[j], ax is the axis handle. Note that this axis
    # will be modified.

    # Note that if an outline is desired, then the outline must not include the entire figure, otherwise an
    # AssertionError is thrown.
    plt.sca(ax)
    if cmap_lims is None:
        sns.heatmap(obs[::-1, :], cmap="coolwarm", xticklabels=np.around(x, decimals=2),
                    yticklabels=np.around(y[::-1], decimals=2), annot=False, vmin=max(np.round(np.amin(obs), 2), 0.0),
                    vmax=np.round(np.amax(obs), 2))
    else:
        sns.heatmap(obs[::-1, :], cmap="coolwarm", xticklabels=np.around(x, decimals=2),
                    yticklabels=np.around(y[::-1], decimals=2), annot=False, vmin=cmap_lims[0],
                    vmax=cmap_lims[1])
    # print np.round(np.amin(obs), 2), np.round(np.amax(obs), 2)
    # for i0 in range(obs.shape[0])
    if outline:
        temp1 = np.transpose(obs[:, :])
        ind2 = 0
        for ind0 in range(temp1.shape[0]):
            for ind1 in range(temp1.shape[1]):
                if val - bound <= temp1[ind0, ind1] <= val + bound:
                    temp2 = LineString(
                        [(ind0, ind1), (ind0 + 1, ind1), (ind0 + 1, ind1 + 1), (ind0, ind1 + 1), (ind0, ind1)])
                    if ind2 == 1:
                        temp3 = temp3.symmetric_difference(temp2)
                    else:
                        temp3 = temp2
                        ind2 += 1
        if ind2 == 1:
            dil = temp3.buffer(0.02)
            patch = PolygonPatch(dil, facecolor=color, edgecolor=color, alpha=alpha)
            ax.add_patch(patch)
    if xlabel:
        # ax.set_xlabel(xlabel, size=20)
        ax.set_xlabel(xlabel)
    if ylabel:
        # ax.set_ylabel(ylabel, size=20)
        ax.set_ylabel(ylabel)
    if title:
        # ax.set_title(title, size=20)
        ax.set_title(title)
    plt.xticks(size=20)
    plt.yticks(size=20)
    ax.tick_params(labelsize=14)
    return ax