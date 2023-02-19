"""Climate predictor - the data extraction part

Responsible for stitching together elevation data and creating more parameters to be used in
learning, such as water influence, elevation differences, and distances from water.

Also contains functions for visualizing array data and turning them into animations

Data for targets from https://www.worldclim.org/data/worldclim21.html

Data for elevation from http://www.viewfinderpanoramas.org/dem3.html

Data for bathymetry from https://www.gebco.net/data_and_products/gridded_bathymetry_data/
- Values range from 0 to 252 (where 255 is brightest?)
- Max depth is 11034 m

Recommended usage:

resolution = (360, 180)
download_inputs_unprocessed(resolution)
download_targets(resolution)
"""

from PIL import Image
import PIL
from scipy.ndimage.filters import gaussian_filter
from typing import Union
from visualize import *

Image.MAX_IMAGE_PIXELS = None


def import_map(file: str, resolution: Optional[tuple[int, int]] = None,
               to_array: bool = False) -> Union[PIL.Image.Image, np.ndarray]:
    """Import and return an equirectangular map as a pillow image (default) or numpy array.

    :param file: The filepath of the image.
    :param resolution: The resolution of the data, aka dimensions of the matrix. By default, chooses
    the original image's resolution
    :param to_array: Whether to return a pillow image or a numpy matrix.
    """
    image = Image.open(f"data/{file}")

    resolution = image.size if resolution is None else resolution
    image = image.resize(resolution, PIL.Image.NEAREST)

    return np.array(image) if to_array else image


def download_inputs_unprocessed(resolution: tuple[int, int] = (360, 180)) -> None:
    """A master function to load raw data, format it properly into input
    data, and download it for future use.

    Downloads elevation data and lake data.
    Coastline data comes from load_land(), which gets its data from target data.

    :param resolution: The resolution to save the data in, aka dimensions of the matrix.
    """
    width, height = resolution
    small_res = (width // 6, height // 4)

    assert width == 2 * height, "Input width is not input height doubled"
    assert small_res == (width / 6, height / 4), "6 & 4 must divide resolution width & height"

    # Get elevation data
    letter = [chr(x) for x in range(65, 89)]
    cells = (import_map(f"elevation/15-{x}.tif", small_res, True).astype('float64') for x in letter)
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x = cells
    elevation = np.r_[np.c_[a, b, c, d, e, f],
                      np.c_[g, h, i, j, k, l],
                      np.c_[m, n, o, p, q, r],
                      np.c_[s, t, u, v, w, x]]
    elevation[elevation < 0] = 0

    # Old way of getting coastline data
    # bathymetry = np.array(import_map("elevation/gebco_2021.tif", resolution).convert("L"))
    # bathymetry = bathymetry.astype('float64') * -11034 / 252

    # Get lake data
    lakes = import_map("elevation/lakes.png", resolution, True)

    # Save data
    np.save(f"data/{width}_by_{height}/elevation", elevation)
    # np.save(f"data/{width}_by_{height}/bathymetry", bathymetry)
    np.save(f"data/{width}_by_{height}/lakes", lakes)
    print(f"Successfully saved raw input data in {width} by {height} resolution")


def import_inputs(resolution: tuple[int, int] = (360, 180), flip_lr: bool = False,
                  flip_ud: bool = False, remove_na: bool = True) -> np.ndarray:
    """Load the input data in the format of a design matrix. Requires unprocessed raw data to be
    downloaded first.

    :param resolution: The resolution of the saved data
    :param flip_lr: Horizontally flip the the world map
    :param flip_ud: Vertically flip the world map
    :param remove_na: Remove rows with na values (ie. water)
    """
    w, h = resolution

    # Load elevation data
    elevation = np.load(f"data/{w}_by_{h}/elevation.npy")
    # Load coastline data
    land = load_land(resolution)

    # Old way of getting coastline data
    # bathymetry = np.load(f"data/{w}_by_{h}/bathymetry.npy")
    # land = (bathymetry == 0).astype("int8")

    if flip_lr:
        elevation = np.fliplr(elevation)
        land = np.fliplr(land)
    if flip_ud:
        elevation = np.flipud(elevation)
        land = np.flipud(land)
    return process_raw_data(land, elevation, remove_na)


def process_raw_data(land: np.ndarray, elevation: np.ndarray,
                     remove_na: bool = False) -> np.ndarray:
    """Process the raw data to obtain all inputs.

    :param land: Matrix data of 0s and 1s outlining what is land (1) and what is water (0)
    :param elevation: Matrix data of elevation
    :param remove_na: Remove rows with na values (ie. water)
    """
    h, w = land.shape
    prop = lambda x: x * (h // 180)  # Scales a number x with resolution

    # Inland, water influence, elevation differences
    inland = inland_distances(land, prop(7))
    water_influence = closest_water_influence(land, prop(10), prop(2))
    elev_diff_3 = elevation_differences(elevation, prop(3)).reshape((8, w * h)).T
    elev_diff_5 = elevation_differences(elevation, prop(5)).reshape((8, w * h)).T
    elev_diff_10 = elevation_differences(elevation, prop(10)).reshape((8, w * h)).T
    elev_diff_15 = elevation_differences(elevation, prop(15)).reshape((8, w * h)).T

    # Latitude
    latitude = get_latitude((w, h))

    # Water distance
    def compare(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Compare the values in x1 and x2, which are both non-negative.
        If x2 > x1, return positive
        If x2 = x1, return 0
        If x2 < x2, return negative
        """
        return np.log((x2 + 0.1) / (x1 + 0.1))

    d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12 = d_context(land)
    d13, d14, d15, d16, d17, d18, d19, d20 = d_no_context(land)
    d21 = compare(d1, d4)
    d22 = compare(d2, d3)
    d23 = compare(d5, d6)
    d24 = compare(d7, d8)
    d25 = compare(d13, d14)
    d26 = compare(d15, d16)
    d27 = compare(d18, d19)
    d28 = compare(d17, d20)

    # Combine inputs
    inland[land == 0] = np.nan
    inputs = np.c_[latitude, elevation.reshape(w * h, 1),
                   inland.reshape(w * h, 1), water_influence.reshape(w * h, 1),
                   elev_diff_3, elev_diff_5, elev_diff_10, elev_diff_15,
                   d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                   d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28]
    return remove_na_rows(inputs) if remove_na else inputs


def d_context(land: np.ndarray) -> tuple:
    """Calculate 8-directional distances from water with water influence context.

    l, r, t, b = left, right, top, bottom
    bl, br, tl, tr = bottom-left, bottom-right, top-left, top-right
    """
    cols = np.product(land.shape)
    d = lambda sides, diag: inland_distances_monodirectional(land, 7, sides, diag).reshape(cols, 1)
    tr = d((False, False, False, True), True)
    tl = d((False, False, True, False), True)
    br = d((False, True, False, False), True)
    bl = d((True, False, False, False), True)
    b = d((False, False, False, True), False)
    t = d((False, False, True, False), False)
    r = d((False, True, False, False), False)
    l = d((True, False, False, False), False)
    br_tl = d((False, True, True, False), True)
    bl_tr = d((True, False, False, True), True)
    l_r = d((True, True, False, False), False)
    t_b = d((False, False, True, True), False)
    return tr, tl, br, bl, b, t, r, l, br_tl, bl_tr, l_r, t_b


def d_no_context(land: np.ndarray) -> tuple:
    """Calculate 8-directional distances from water with no water influence context.

    l, r, t, b = left, right, top, bottom
    bl, br, tl, tr = bottom-left, bottom-right, top-left, top-right
    """
    cols = np.product(land.shape)
    dd = lambda sides, diag: find_coastline_distances(land, sides, diag).reshape(cols, 1)
    l = dd((True, False, False, False), False)
    r = dd((False, True, False, False), False)
    t = dd((False, False, True, False), False)
    b = dd((False, False, False, True), False)
    bl = dd((True, False, False, False), True)
    br = dd((False, True, False, False), True)
    tl = dd((False, False, True, False), True)
    tr = dd((False, False, False, True), True)
    return l, r, t, b, bl, br, tl, tr


def remove_na_rows(m: np.ndarray) -> np.ndarray:
    """Remove rows that contain a na value."""
    return m[~np.isnan(m).any(axis=1)]


def remove_na_cols(m: np.ndarray) -> np.ndarray:
    """Remove cols that contain a na value."""
    return m[:, ~np.any(np.isnan(m), axis=0)]


def load_land(resolution: tuple[int, int] = (360, 180)) -> np.ndarray:
    """Load the boundary for separating land and water. Must be all 0s or 1s."""
    w, h = resolution
    temp_avg = np.load(f"data/{w}_by_{h}/temp_avg.npy")[0]
    lakes = np.load(f"data/{w}_by_{h}/lakes.npy")
    land = np.logical_and(lakes == 0, ~np.isnan(temp_avg))
    return land


####################################################################################################
# Processing to obtain more inputs
####################################################################################################
def get_latitude(resolution: tuple[int, int] = (360, 180)) -> np.ndarray:
    """Return latitude."""
    w, h = resolution
    prop = lambda x: x * (h // 180)

    cols = np.product(resolution)
    # x = np.repeat(np.array(range(w)).reshape(1, w), h, axis=0).reshape(cols, 1)
    y = np.repeat(np.array(range(h)).reshape(h, 1), w, axis=1).reshape(cols, 1)
    latitude = 90 - y - prop(0.5)
    return latitude


def shuffle(m: np.ndarray, direction: str, i: int = 1) -> np.ndarray:
    """Shuffle a matrix in one direction by i pixels .

    Horizontal shuffling preserves original columns, while vertical shuffling does not.
    """
    assert direction in {"top", "bottom", "left", "right"}
    if i == 0:
        return m
    elif direction == "left":
        return np.c_[m[:, i:], m[:, :i]]
    elif direction == "right":
        return np.c_[m[:, -i:], m[:, :-i]]
    elif direction == "top":
        return np.r_[m[i:, :], m[-1, :].reshape(1, m.shape[1]).repeat(i, axis=0)]
    else:
        return np.r_[m[0, :].reshape(1, m.shape[1]).repeat(i, axis=0), m[:-i, :]]


def find_coastlines_straight(land: np.ndarray, hard_edge: bool = True,
                             sides: tuple[bool, bool, bool, bool] = (
                                     True, True, True, True)) -> np.ndarray:
    """Find the coastlines in a world map of 1s and 0s using straight distance."""
    assert sum(sides) >= 1

    l, r, t, b = sides
    left = shuffle(land, "left") if l else 0
    right = shuffle(land, "right") if r else 0
    top = shuffle(land, "top") if t else 0
    bottom = shuffle(land, "bottom") if b else 0

    edges = (l + r + t + b) * land - left - right - top - bottom
    return np.logical_and(edges != 0, land == 1).astype("int8") if hard_edge else edges


def find_coastlines_diagonal(land: np.ndarray, hard_edge: bool = True,
                             sides: tuple[bool, bool, bool, bool] = (
                                     True, True, True, True)) -> np.ndarray:
    """Find the coastlines in a world map of 1s and 0s using diagonal distance."""
    assert sum(sides) >= 1

    bl, br, tl, tr = sides
    bottom_left = shuffle(shuffle(land, "left"), "bottom") if bl else 0
    bottom_right = shuffle(shuffle(land, "right"), "bottom") if br else 0
    top_left = shuffle(shuffle(land, "left"), "top") if tl else 0
    top_right = shuffle(shuffle(land, "right"), "top") if tr else 0

    edges = (bl + br + tl + tr) * land - bottom_left - bottom_right - top_left - top_right
    return np.logical_and(edges != 0, land == 1).astype("int8") if hard_edge else edges


def find_coastlines(land: np.ndarray, hard_edge: bool = True,
                    sides: tuple[bool, bool, bool, bool] = (True, True, True, True),
                    diagonal: bool = False) -> np.ndarray:
    """Find the coastlines in a world map of 1s and 0s."""
    if diagonal:
        return find_coastlines_diagonal(land, hard_edge, sides)
    else:
        return find_coastlines_straight(land, hard_edge, sides)


def find_coastline_distances(land: np.ndarray,
                             sides: tuple[bool, bool, bool, bool] = (True, True, True, True),
                             diagonal: bool = False) -> np.ndarray:
    """Find how many pixels away each point to a coastline in a world map of 1s and 0s."""
    l, distances = land.copy(), np.zeros(land.shape).astype("float64")

    ticker = 0
    while not np.all(l == 0) and ticker <= l.shape[1]:
        edges = find_coastlines(l, True, sides, diagonal)
        land_outer = np.logical_and(l == 1, edges != 0)
        land_inner = np.logical_and(l == 1, edges == 0)
        l[land_outer] = 0
        distances[land_inner] += 1
        ticker += 1
    return distances


def find_smooth_coastline_distances(land: np.ndarray) -> np.ndarray:
    """Find how many pixels away each point to a coastline in a world map of 1s and 0s."""
    straight = find_coastline_distances(land, diagonal=False)
    diagonal = find_coastline_distances(land, diagonal=True) * (2 ** 0.5) / 2
    return 0.5 * (straight + diagonal)


def inland_distances(land: np.ndarray, i: int = 7) -> np.ndarray:
    """Find how many pixels away each point to a coastline in a world map of 1s and 0s.
    More sensitive to land size.

    i is the averaging factor.

    Ideal value at (360, 180) resolution is 7 on land, 10 at sea
    """
    l, water = land.copy(), 1 - land
    coastlines = []
    for _ in range(i):
        coastlines.append(find_smooth_coastline_distances(l))
        edges = find_coastlines(water, True)
        water_outer = np.logical_and(water == 1, edges != 0)
        water[water_outer] = 0
        l = 1 - water
    return np.multiply(sum(coastlines) / i, land)


def inland_distances_monodirectional(land: np.ndarray, i: int = 7,
                                     sides: tuple[bool, bool, bool, bool] = (True, True, True, True),
                                     diagonal: bool = False) -> np.ndarray:
    """Find how many pixels away each point to a coastline in a world map of 1s and 0s.
    More sensitive to land size.

    i is the averaging factor.

    Ideal value at (360, 180) resolution is 7 on land, 10 at sea
    """
    l, water = land.copy(), 1 - land
    coastlines = []
    for _ in range(i):
        coastlines.append(find_coastline_distances(l, sides, diagonal))
        edges = find_coastlines(water, True)
        water_outer = np.logical_and(water == 1, edges != 0)
        water[water_outer] = 0
        l = 1 - water
    return np.multiply(sum(coastlines) / i, land)


def elevation_differences(elevation: np.ndarray, radius: float = 10.) -> np.ndarray:
    """Calculate elevation differences in a circle of given radius."""
    z = np.zeros(elevation.shape)
    directions = np.array([z.copy(), z.copy(), z.copy(), z.copy(),
                           z.copy(), z.copy(), z.copy(), z.copy()])
    # counter-clockwise, starting at right aka 0 degrees
    normalizer = np.array([0., 0., 0., 0., 0., 0., 0., 0.])

    for dist_h in {"left", "right"}:
        i = 0
        while i <= radius:
            for dist_v in {"top", "bottom"}:
                j = 0 if i != 0 else 1
                while i ** 2 + j ** 2 <= radius ** 2:
                    elevation_offset = shuffle(shuffle(elevation, dist_h, i), dist_v, j)

                    ii = i if dist_h == "right" else -i
                    jj = -j if dist_v == "top" else j
                    angle = coordinate_angle(ii, jj)

                    weights = [max(0., 1 - abs(angle / 22.5), 1 - abs((angle - 360) / 22.5))]
                    weights += [max(0., 1 - abs((angle - x) / 22.5)) for x in range(45, 360, 45)]

                    directions_offset = np.array([(elevation_offset - elevation) * x for x in weights])
                    directions += directions_offset
                    normalizer += weights
                    j += 1
            i += 1
    return directions / normalizer.reshape((8, 1, 1))


def coordinate_angle(x: int, y: int) -> float:
    """Calculate the angle of a coordinate with respect to the origin, in degrees."""
    assert x != 0 or y != 0
    if x == 0:
        return 90 if y > 0 else 270
    degrees = np.arctan(y / x) * 180 / np.pi
    if x < 0:
        offset = 180
    elif y >= 0:
        offset = 0
    else:
        offset = 360
    return degrees + offset


def closest_water_influence(land: np.ndarray, i: int = 20, sigma: float = 2.) -> np.ndarray:
    """Find how much influence the closest water body to land has."""
    l, water = land.copy(), 1 - land
    water_influence = inland_distances(water, i)
    edges = find_coastlines(l)

    while not np.all(edges == 0):
        inner = inner_layer(water_influence) * edges
        water_influence += inner
        l[inner != 0] = 0
        edges = find_coastlines(l)

    water_influence = np.c_[water_influence, water_influence[:, :6 * sigma]]
    water_influence = gaussian_filter(water_influence, sigma=sigma)
    water_influence = np.c_[water_influence[:, water.shape[1]: water.shape[1] + 3 * sigma],
                            water_influence[:, 3 * sigma: water.shape[1]]]
    return water_influence


# def find_coastline_distances_v2(land: np.ndarray, sides: tuple[bool, bool] = (True, True),
#                                 diagonal: bool = False) -> np.ndarray:
#     """Find how many pixels away each point to a coastline in a world map of 1s and 0s.
#     More sensitive to land size."""
#     l, distances = land.copy(), np.zeros(land.shape).astype("float64")
#
#     edges = find_coastlines(l, False, sides, diagonal)
#     while not np.all(edges == 0):
#         positive_edges = np.logical_and(land == 1, edges > 0)
#         l[positive_edges] = 0
#         distances += np.multiply(positive_edges, np.e ** -edges)
#         inner_offset = inner_layer(distances)
#         distances += np.multiply(inner_offset, np.logical_and(l == 1, positive_edges == 0))
#
#         edges = find_coastlines(l, False, sides, diagonal)
#         visualize(distances)
#     return distances


def inner_layer(distances: np.ndarray) -> np.ndarray:
    """Return a prediction of the outermost layer's value, based on the values of outside layers."""
    top = shuffle(distances, "top")
    bottom = shuffle(distances, "bottom")
    left = shuffle(distances, "left")
    right = shuffle(distances, "right")
    vertical = top + bottom
    vertical[np.logical_and(top != 0, bottom != 0)] /= 2
    horizontal = left + right
    horizontal[np.logical_and(left != 0, right != 0)] /= 2
    result = vertical + horizontal
    result[np.logical_and(vertical != 0, horizontal != 0)] /= 2
    return result


####################################################################################################
# Targets
####################################################################################################
def download_targets(resolution: tuple[int, int] = (360, 180)) -> None:
    """Load the target data, format it, and save it.

    :param resolution: The resolution to store the data in
    """
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    import_target = lambda p1, p2: [import_map(f"{p1}/wc2.1_5m_{p2}_{x}.tif",
                                               resolution, True).astype('float64') for x in months]

    temp_avg = import_target("temperature_average", "tavg")
    temp_max = import_target("temperature_max", "tmax")
    temp_min = import_target("temperature_min", "tmin")
    precipitation = import_target("precipitation", "prec")
    sunlight = import_target("solar_radiation", "srad")
    vapour = import_target("water_vapour_pressure", "vapr")
    wind = import_target("wind_speed", "wind")

    for i in range(12):
        for variable in [temp_avg, temp_max, temp_min, precipitation, sunlight, vapour, wind]:
            variable[i][np.logical_or(variable[i] <= -32768, variable[i] >= 65535)] = np.nan
    w, h = resolution

    np.save(f"data/{w}_by_{h}/temp_avg", temp_avg)
    np.save(f"data/{w}_by_{h}/temp_max", temp_max)
    np.save(f"data/{w}_by_{h}/temp_min", temp_min)
    np.save(f"data/{w}_by_{h}/precipitation", precipitation)
    np.save(f"data/{w}_by_{h}/sunlight", sunlight)
    np.save(f"data/{w}_by_{h}/vapour", vapour)
    np.save(f"data/{w}_by_{h}/wind", wind)
    print(f"Successfully saved raw target data in {w} by {h} resolution")


def import_targets(resolution: tuple[int, int] = (360, 180), flip_lr: bool = False,
                   flip_ud: bool = False, remove_na: bool = True) -> np.ndarray:
    """Load the target data in the format suitable for machine learning.

    :param resolution: The resolution of the stored data to access
    :param flip_lr: Flip the matrix horizontally
    :param flip_ud: Flip the matrix vertically
    :param remove_na: Remove rows with na values (ie. water)
    """
    w, h = resolution
    temp_avg = np.load(f"data/{w}_by_{h}/temp_avg.npy")
    temp_max = np.load(f"data/{w}_by_{h}/temp_max.npy")
    temp_min = np.load(f"data/{w}_by_{h}/temp_min.npy")
    precipitation = np.load(f"data/{w}_by_{h}/precipitation.npy")
    sunlight = np.load(f"data/{w}_by_{h}/sunlight.npy")
    vapour = np.load(f"data/{w}_by_{h}/vapour.npy")
    wind = np.load(f"data/{w}_by_{h}/wind.npy")

    land = load_land(resolution)
    for i in range(12):
        for variable in [temp_avg, temp_max, temp_min, precipitation, sunlight, vapour, wind]:
            variable[i][land == 0] = np.nan

    fliplr = lambda x: np.fliplr(x) if flip_lr else x
    flipud = lambda x: np.flipud(x) if flip_ud else x
    removena = lambda x: remove_na_cols(x) if remove_na else x
    reshaping = lambda x: removena(fliplr(flipud(x.reshape((12, w * h)))))
    temp_avg = reshaping(temp_avg)
    temp_max = reshaping(temp_max)
    temp_min = reshaping(temp_min)
    precipitation = reshaping(precipitation)
    sunlight = reshaping(sunlight)
    vapour = reshaping(vapour)
    wind = reshaping(wind)

    targets = np.array([temp_avg, temp_max, temp_min, precipitation, sunlight, vapour, wind])
    return targets
