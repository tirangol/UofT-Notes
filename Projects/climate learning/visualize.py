"""Climate predictor - the visualization part

Responsible for visualizing maps and turning them into animated gifs.
Responsible for computing Koppen climate groups and visualizing them.

https://hess.copernicus.org/articles/11/1633/2007/hess-11-1633-2007.pdf
https://opus.bibliothek.uni-augsburg.de/opus4/frontdoor/deliver/index/docId/40083/file/metz_Vol_15_No_3_p259-263_World_Map_of_the_Koppen_Geiger_climate_classification_updated_55034.pdf

"""
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from typing import Optional, Callable


def visualize(matrix: np.ndarray, limits: Optional[tuple[int, int]] = None) -> None:
    """Visualize a matrix as a heat map."""
    plt.clf()
    if limits is None:
        plt.imshow(matrix)
    else:
        plt.imshow(matrix, vmin=limits[0], vmax=limits[1])
    plt.colorbar()
    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()


def animate(matrix: np.ndarray, name: str, resolution: tuple[int, int] = (360, 180),
            limits: Optional[tuple[int, int]] = None) -> None:
    """Create a gif of the matrix, counting each column as a frame.

    The shape of matrix should be ... x 12, where ... is a very big number.
    """
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    _, frames = matrix.shape
    w, h = resolution

    # Create images from each frame and save them
    for i in range(frames):
        data = matrix.T[i].reshape(h, w)
        plt.clf()
        plt.title(f"Month {i + 1} ({months[i]})")
        if limits is None:
            plt.imshow(data)
        else:
            plt.imshow(data, vmin=limits[0], vmax=limits[1])
        plt.colorbar()
        plt.get_current_fig_manager().window.state('zoomed')
        plt.savefig(f'img/{name}_{i}.png')

    # Create gif and delete each frame
    with imageio.get_writer(f'img/{name}.gif', mode="I") as writer:
        for i in range(frames):
            image = imageio.v2.imread(f"img/{name}_{i}.png")
            writer.append_data(image)
            os.remove(f"img/{name}_{i}.png")
    print(f"Successfully created gif at img/{name}.gif")


def koppen(avg_temp: np.ndarray, prec: np.ndarray, latitude: np.ndarray, legend: int = 1) -> None:
    """Return a map of koppen climate indicator values.

    :param avg_temp: A 12 by __ (very big) array of monthly average temperature data
    :param prec: A 12 by __ (very big) array of monthly precipitation
    :param latitude: An 1 by ___ (very big) array of latitude
    :param legend: Information about the legend. 0 for do not display, 1 for display, 2 for verbose.
    """
    latitude = latitude.reshape(avg_temp.shape[1])

    def hemisphere(north: bool) -> np.ndarray:
        """Return a hemisphere."""
        return (latitude >= 0) if north else (latitude < 0)

    def apr_sep(x: np.ndarray, op: Callable, hem: bool) -> np.ndarray:
        """Return data in x, aggregrated with numpy operation op, from April to September
        (Summer half), in hemisphere hem."""
        return op(x[3:9], axis=0) * hemisphere(hem)

    def oct_mar(x: np.ndarray, op: Callable, hem: bool) -> np.ndarray:
        """Return data in x, aggregrated with numpy operation op, from October to March
        (Winter half), in hemisphere hem."""
        return op((op(x[9:], axis=0), op(x[:3], axis=0)), axis=0) * hemisphere(hem)

    # A-type climates
    hottest = np.nanmax(avg_temp, axis=0)
    coldest = np.nanmin(avg_temp, axis=0)
    driest = np.nanmin(prec, axis=0)
    annual_prec = np.nansum(prec, axis=0)
    annual_avg_temp = np.nansum(avg_temp, axis=0) / 12

    A = coldest >= 18
    Af = A * (driest > 60)
    Am = A * (driest <= 60) * (driest >= 100 - (annual_prec / 25))
    Aw = A * (driest < 100 - (annual_prec / 25))

    # B-type climates
    warm_prec = apr_sep(prec, np.nansum, True) + oct_mar(prec, np.nansum, False)
    warm_prec_percent = (warm_prec + 0.001) / (annual_prec + 0.001)
    t1 = warm_prec_percent >= 0.7
    t2 = (warm_prec_percent < 0.7) * (warm_prec_percent >= 0.3)
    t3 = 1 - t1 - t2
    threshold = t1 * (2 * annual_avg_temp + 28) + t2 * (2 * annual_avg_temp + 14) + t3 * (2 * annual_avg_temp)

    B = (hottest >= 10) * (annual_prec < 10 * threshold)
    BW = (annual_prec < 5 * threshold) * B
    BS = (annual_prec >= 5 * threshold) * B
    BWh = BW * (annual_avg_temp >= 18)
    BWk = BW * (annual_avg_temp < 18)
    BSh = BS * (annual_avg_temp >= 18)
    BSk = BS * (annual_avg_temp < 18)

    # C-type climates
    driest_summer = apr_sep(prec, np.nanmin, True) + oct_mar(prec, np.nanmin, False)
    driest_winter = oct_mar(prec, np.nanmin, True) + apr_sep(prec, np.nanmin, False)
    wettest_summer = apr_sep(prec, np.nanmax, True) + oct_mar(prec, np.nanmax, False)
    wettest_winter = oct_mar(prec, np.nanmax, True) + apr_sep(prec, np.nanmax, False)

    s = (driest_summer < 40) * (wettest_summer < wettest_winter / 3)
    w = (driest_winter < wettest_summer / 10)

    above_10 = np.nansum(avg_temp > 10, axis=0)
    a = (hottest >= 22)
    b = (hottest < 22) * (above_10 >= 4)
    c = (hottest < 22) * (above_10 < 4) * (above_10 >= 1)

    C = (hottest >= 10) * (0 < coldest) * (coldest < 18)
    Cs, Cw = C * s, C * w
    Cf = np.logical_xor(C, np.maximum(Cs, Cw))
    Csa, Cwa, Cfa = Cs * a, Cw * a, Cf * a
    Csb, Cwb, Cfb = Cs * b, Cw * b, Cf * b
    Csc, Cwc, Cfc = Cs * c, Cw * c, Cf * c

    # D-type climates
    d = (hottest < 22) * (above_10 < 4) * (coldest < -38)
    c = np.logical_not(np.logical_or.reduce((a, b, d)))

    D = (hottest >= 10) * (coldest <= 0)
    Ds, Dw = D * s, D * w
    Df = np.logical_xor(D, np.maximum(Ds, Dw))
    Dsa, Dwa, Dfa = Ds * a, Dw * a, Df * a
    Dsb, Dwb, Dfb = Ds * b, Dw * b, Df * b
    Dsd, Dwd, Dfd = Ds * d, Dw * d, Df * d
    Dsc, Dwc, Dfc = Ds * c, Dw * c, Df * c

    # E-type climates
    E = hottest < 10
    ET = E * (hottest > 0)
    EF = E * (hottest <= 0)

    # Simplified visualization of A, B, C, D, E
    # k = np.empty(avg_temp.shape[1])
    # k[C] = 2
    # k[A] = 4
    # k[D] = 1
    # k[B] = 3
    # k[E] = 0
    # water = np.isnan(prec[0].reshape(np.product(prec[0].shape)))
    # k[water] = np.nan
    #
    # cmap = colors.ListedColormap(['lightslategray', 'darkgreen', 'limegreen', 'khaki', 'forestgreen'])
    # bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    # norm = colors.BoundaryNorm(bounds, cmap.N)
    # plt.imshow(k.reshape(180, 360), cmap=cmap, norm=norm)

    # Visualization of all categories
    # Create the map k, set values in k to different numbers depending on category
    k = np.empty(avg_temp.shape[1])
    k[Cfa], k[Cfb], k[Cfc], k[Csa], k[Csb], k[Csc], k[Cwa], k[Cwb], k[Cwc] = range(7, 16)
    k[Af], k[Am], k[Aw] = range(0, 3)
    k[Dfa], k[Dfb], k[Dfc], k[Dfd] = range(16, 20)
    k[Dsa], k[Dsb], k[Dsc], k[Dsd] = range(20, 24)
    k[Dwa], k[Dwb], k[Dwc], k[Dwd] = range(24, 28)
    k[BSh], k[BSk], k[BWh], k[BWk] = range(3, 7)
    k[EF] = 28
    k[ET] = 29
    k[np.isnan(prec[0].reshape(np.product(prec[0].shape)))] = np.nan

    # Create colours for each possible number
    rgb = [(0, 0, 255), (0, 120, 255), (70, 170, 250),
           (245, 165, 0), (255, 220, 100), (255, 0, 0), (255, 150, 150),
           (200, 255, 80), (100, 255, 80), (50, 200, 0),
           (255, 255, 0), (200, 200, 0), (150, 150, 0),
           (150, 255, 150), (100, 200, 100), (50, 150, 50),
           (0, 255, 255), (55, 200, 255), (0, 125, 125), (0, 70, 95),
           (255, 0, 255), (200, 0, 200), (150, 50, 150), (150, 100, 150),
           (170, 175, 255), (90, 120, 220), (75, 80, 180), (50, 0, 135),
           (102, 102, 102), (178, 178, 178)]
    rgb = [(x / 255, y / 255, z / 255) for x, y, z in rgb]

    cmap = colors.ListedColormap(rgb)
    bounds = np.arange(start=-0.5, stop=30.5, step=1)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # Create labels for each possible number
    labels_verbose = ["Af (Tropical rainforest)",
                      "Am (Tropical monsoon)",
                      "Aw/As (Tropical savanna)",
                      "BSh (Hot semi-arid)",
                      "BSk (Cold semi-arid)",
                      "BWh (Hot desert)",
                      "Bwk (Cold desert)",
                      "Cfa (Humid subtropical)",
                      "Cfb (Temperate oceanic)",
                      "Cfc (Subpolar oceanic)",
                      "Csa (Hot-summer Mediterranean)",
                      "Csb (Warm-summer Mediterranean)",
                      "Csc (Cold-summer Mediterranean)",
                      "Cwa (Humid subtropical monsoon)",
                      "Cwb (Temperate monsoon)",
                      "Cwc (Subpolar oceanic monsoon)",
                      "Dfa (Hot-summer humid continental)",
                      "Dfb (Warm-summer humid continental)",
                      "Dfc (Subarctic)",
                      "Dfd (Extremely-cold subarctic)",
                      "Dsa (Hot-summer Mediterranean humid continental)",
                      "Dsb (Warm-summer Mediterranean humid continental)",
                      "Dsc (Subarctic Mediterranean)",
                      "Dsd (Extremely-cold Mediterranean arctic)",
                      "Dwa (Hot-summer monsoon humid continental)",
                      "Dwb (Warm-summer monsoon humid continental)",
                      "Dwc (Subarctic monsoon)",
                      "Dwd (Extremely-cold monsoon arctic)",
                      "EF (Ice cap)",
                      "ET (Tundra)"]
    labels_concise = [x[:x.find("(") - 1] for x in labels_verbose]

    # Plot data
    plt.clf()
    plt.imshow(k.reshape(180, 360), cmap=cmap, norm=norm)
    if legend > 0:
        bar = plt.colorbar(ticks=range(0, 30))
        bar.ax.set_yticklabels(labels_concise if legend == 1 else labels_verbose)
