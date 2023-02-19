"""Climate predictor - the data processing part

Responsible for processing extracted parameters from last part into features that are more helpful
and conducive to machine learning.

Recommended usage:

resolution = (360, 180)
save_inputs(resolution, False, (False, False, False))
save_inputs(resolution, False, (True, True, False))
save_inputs(resolution, False, (True, False, False))
save_inputs(resolution, False, (False, True, False))
save_inputs(resolution, True, (True, True, True))

save_targets(resolution, False, (False, False, False))
save_targets(resolution, False, (True, True, False))
save_targets(resolution, False, (True, False, False))
save_targets(resolution, False, (False, True, False))
save_targets(resolution, True, (True, True, True))

save_temp_inputs(resolution, False, (False, False, False))
save_temp_inputs(resolution, False, (False, True, False))
save_temp_inputs(resolution, False, (True, False, False))
save_temp_inputs(resolution, False, (True, True, False))
save_temp_inputs(resolution, True, (True, True, True))
"""

from load_data import *
import torch
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from typing import Callable


####################################################################################################
# Load input data
####################################################################################################
def format_filepath(w: int, h: int, remove_na: bool,
                    flip: tuple[bool, bool, bool], s: str) -> str:
    """Format the file-path to an input/target."""
    f1, f2, f3 = flip
    p1 = '' if remove_na else '_na'
    if f3:
        p2 = '_flipped_combined'
    elif f1 and f2:
        p2 = '_flipped_all'
    elif f1:
        p2 = '_flipped_lr'
    elif f2:
        p2 = '_flipped_ud'
    else:
        p2 = ''
    return f"data/{w}_by_{h}/learning/{s}{p1}{p2}.npy"


def save_inputs(resolution: tuple[int, int], remove_na: bool = False,
                flip: tuple[bool, bool, bool] = (False, False, False)) -> None:
    """Download all inputs in a specified format. Requires raw inputs to be downloaded.

    :param resolution: The resolution of the raw inputs to load
    :param remove_na: Omit na values (ie. remove water)
    :param flip: A tuple of flipping parameters for flipping horizontally, vertically, or to return
    data for non-flipped and all-flipped.
    """
    w, h = resolution
    f1, f2, f3 = flip
    if f3:
        i = np.r_[import_inputs(resolution, False, False, remove_na),
                  import_inputs(resolution, True, True, remove_na)]
    else:
        i = import_inputs(resolution, f1, f2, remove_na)

    np.save(format_filepath(w, h, remove_na, flip, "inputs"), i)
    print(f"Successfully saved inputs in {w} by {h} resolution, "
          f"remove_na={remove_na}, flip={flip}")


def save_targets(resolution: tuple[int, int], remove_na: bool = False,
                 flip: tuple[bool, bool, bool] = (False, False, False)) -> None:
    """Download all targets in a specified format. Requires raw targets to be downloaded.

    :param resolution: The resolution of the raw targets to load
    :param remove_na: Omit na values (ie. remove water)
    :param flip: A tuple of flipping parameters for flipping horizontally, vertically, or to return
    data for non-flipped and all-flipped.
    """
    w, h = resolution
    f1, f2, f3 = flip
    if f3:
        t = np.concatenate((import_targets(resolution, False, False, remove_na),
                            import_targets(resolution, True, True, remove_na)), axis=2)
    else:
        t = import_targets(resolution, f1, f2, remove_na)

    np.save(format_filepath(w, h, remove_na, flip, "targets"), t)
    print(f"Successfully saved targets in {w} by {h} resolution, "
          f"remove_na={remove_na}, flip={flip}")


def get_inputs(resolution: tuple[int, int], remove_na: bool = False,
               flip: tuple[bool, bool, bool] = (False, False, False)) -> np.ndarray:
    """Retrieve the saved inputs.

    :param resolution: The resolution of the inputs to load
    :param remove_na: Omit na values (ie. remove water)
    :param flip: A tuple of flipping parameters for flipping horizontally, vertically, or to return
    data for non-flipped and all-flipped.
    """
    w, h = resolution
    return np.load(format_filepath(w, h, remove_na, flip, "inputs"))


def get_targets(resolution: tuple[int, int], remove_na: bool = False,
                flip: tuple[bool, bool, bool] = (False, False, False)) -> np.ndarray:
    """Retrieve the saved targets.

    :param resolution: The resolution of the targets to load
    :param remove_na: Omit na values (ie. remove water)
    :param flip: A tuple of flipping parameters for flipping horizontally, vertically, or to return
    data for non-flipped and all-flipped.
    """
    w, h = resolution
    return np.load(format_filepath(w, h, remove_na, flip, "targets"))


####################################################################################################
# Tensors and numpy arrays
####################################################################################################
def to_tensor(m: np.ndarray) -> torch.Tensor:
    """Convert a numpy array map as a tensor."""
    return torch.from_numpy(m).float()


def to_array(m: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a numpy array map."""
    return m.detach().numpy()


####################################################################################################
# Numpy matrix operations
####################################################################################################
def maxes(*matrices: np.ndarray) -> np.ndarray:
    """Return the element-wise maxima of many numpy arrays."""
    assert len(matrices) >= 2
    m = matrices[0]
    for matrix in matrices[1:]:
        m = np.maximum(m, matrix)
    return m


def avgs(*matrices: np.ndarray) -> np.ndarray:
    """Return the element-wise average of many numpy arrays."""
    n = len(matrices)
    assert n >= 2
    m = matrices[0]
    for matrix in matrices[1:]:
        m += matrix
    return m / n


####################################################################################################
# Adjusting/normalizing/balancing matrix values
####################################################################################################
def spread(x: Union[float, np.ndarray, torch.Tensor],
           n: float = 0.5) -> Union[np.ndarray, torch.Tensor, float]:
    """Positively skew data from -1 to 1 that is too biased to -1."""
    return 2 * ((0.5 * (x + 1)) ** n) - 1


def normalize(x: Union[float, np.ndarray, torch.Tensor],
              nan: bool = True) -> Union[np.ndarray, float, torch.Tensor]:
    """Divide x by max(abs(x)), so the data falls within the range of [-1, 1].
    """
    return x / np.nanmax(abs(x)) if nan else x / torch.max(torch.from_numpy(np.array([abs(x)])))


def sigmoid(x: Union[float, np.ndarray], n: float = 0) -> Union[np.ndarray, float]:
    """Apply the sigmoid function on data so it falls within the range [0, 1].

    Treats high-magnitude values as less important.
    """
    return 1 / (1 + np.e ** (-x + n))


def rev_sigmoid(x: Union[float, np.ndarray], n: float = 0) -> Union[np.ndarray, float]:
    """Apply the reverse sigmoid function on data so it falls within the range [0, 1].

    Treats high-magnitude values as less important.
    """
    return 1 - sigmoid(x, n)


def tanh(x: Union[float, np.ndarray], n: float = 1.) -> Union[np.ndarray, float]:
    """Apply the tanh function on data so it falls within the range [-n, n].

    Treats high-magnitude values as less important.
    """
    return n * np.tanh(x / n)


def rev_tanh(x: Union[float, np.ndarray], n: float = 1.) -> Union[np.ndarray, float]:
    """Apply the reverse tanh function on data so it falls within the range [-n, n].

    Treats high-magnitude values as less important.
    """
    return 1 - tanh(x, n)


def relu(x: Union[float, np.ndarray], n: float = 0) -> Union[np.ndarray, float]:
    """Apply the relu function on data so it falls within the range [0, infty)."""
    return np.maximum(x, n)


def rev_relu(x: Union[float, np.ndarray], n: float = 0) -> Union[np.ndarray, float]:
    """Apply the negative relu function on data so it falls within the range (-infty, 0]."""
    return np.minimum(x, n)


####################################################################################################
# Helper functions
####################################################################################################
def straight_coast(prop_s: np.ndarray, dist_s: np.ndarray, prop_i: np.ndarray, dist_i: np.ndarray,
                   water_inf: np.ndarray, d: str = "west") -> tuple[np.ndarray, np.ndarray]:
    """Calculate the influence of west/east coast.

    :param prop_s: The proportion-based, water-sensitive coast distance map
    :param dist_s: The distance-based, water-insensitive coast distance map
    :param prop_i: The proportion_based, water-sensitive coast distance map
    :param dist_i: The proportion_based, water-insensitive coast distance map
    :param water_inf: The water influence map
    :param d: Either "west" or "east"

    We usually have:
    Variable        West Coast          East Coast
    prop_s          d24                 d24
    dist_s          d7                  d8
    prop_i          d25                 d25
    dist_i          d14                 d13
    """
    # rev_normalize = lambda x: (1 - normalize(x))

    # l5_1 = sigmoid(d24, 1) ** 2
    # l5_2 = rev_normalize(d7) ** 20
    # l5_3 = rev_sigmoid(gaussian_filter(d25, prop(2)), 1) ** 2
    # l5_4 = rev_normalize(gaussian_filter(d14, prop(2))) ** 10
    # l5a = avgs(l5_1, l5_2, l5_3, l5_4) * l4
    # l5b = avgs(l5_1, l5_2, l5_3, l5_4) ** 2 * l4

    # l8_1 = rev_sigmoid(d24, 1) ** 2
    # l8_2 = rev_normalize(d8) ** 20
    # l8_3 = sigmoid(gaussian_filter(d25, prop(2)), 1) ** 2
    # l8_4 = rev_normalize(gaussian_filter(d13, prop(2))) ** 10
    # l8a = avgs(l8_1, l8_2, l8_3, l8_4) * l4
    # l8b = avgs(l8_1, l8_2, l8_3, l8_4) ** 2 * l4
    assert d in {"west", "east"}
    prop = lambda x: x * dist_i.shape[0] // 180
    sig = sigmoid if d == "west" else rev_sigmoid
    rev_sig = rev_sigmoid if d == "west" else sigmoid

    p1 = sig(prop_s, 1) ** 2
    p2 = rev_tanh(dist_s / prop(20)) ** 2
    p3 = rev_sig(gaussian_filter(prop_i, 2), 1) ** 2
    p4 = rev_tanh(gaussian_filter(dist_i, 2) / prop(20)) ** 2

    c1 = avgs(p1, p2, p3, p4) * water_inf
    c2 = avgs(p1, p2, p3, p4) ** 2 * water_inf
    return c1, c2


def diagonal_coast(prop_s: np.ndarray, dist_s: np.ndarray, prop_i: np.ndarray, dist_i: np.ndarray,
                   inlandness: Callable, d: str = "south") -> np.ndarray:
    """Calculate the influence of diagonal coasts.

    :param prop_s: The proportion-based, water-sensitive coast distance map
    :param dist_s: The distance-based, water-insensitive coast distance map
    :param prop_i: The proportion_based, water-sensitive coast distance map
    :param dist_i: The proportion_based, water-insensitive coast distance map
    :param inlandness: A function of codomain [0, 1] that returns lower values for inland areas
    :param d: Either "south" or "north"

    We usually have:
    Variable        Southwest Coast     Southeast Coast     Northwest Coast     Northeast Coast
    prop_s          d21                 d22                 d22                 d21
    dist_s          d1                  d2                  d3                  d4
    prop_i          d28                 d27                 d27                 d28
    dist_i          d20                 d19                 d18                 d17
    """
    # l6_1 = rev_sigmoid(gaussian_filter(d22, prop(1)), 1) ** 5
    # l6_2 = rev_normalize(gaussian_filter(tanh(d3, prop(50)), prop(1))) ** 5
    # l6_3 = sigmoid(gaussian_filter(d27, prop(1)), 1) ** 3
    # l6_4 = rev_normalize(gaussian_filter(tanh(d18, prop(50)), prop(1))) ** 5 * inland_weight(1)
    # l6 = avgs(l6_1, l6_2, l6_3, l6_4) * l4 * inland_weight(2)  # Top-left

    # l7_1 = sigmoid(gaussian_filter(d21, prop(1)), 1) ** 5
    # l7_2 = rev_normalize(gaussian_filter(tanh(d1, prop(50)), prop(1))) ** 5
    # l7_3 = rev_sigmoid(gaussian_filter(d28, prop(1)), 1) ** 3
    # l7_4 = rev_normalize(gaussian_filter(tanh(d20, prop(50)), prop(1))) ** 5 * inland_weight(1)
    # l7 = avgs(l7_1, l7_2, l7_3, l7_4) * l4 * inland_weight(2)  # Bottom-left

    # l9_1 = rev_sigmoid(gaussian_filter(d21, prop(1)), 1) ** 5
    # l9_2 = rev_normalize(gaussian_filter(tanh(d4, prop(50)), prop(1))) ** 5
    # l9_3 = sigmoid(gaussian_filter(d28, prop(1)), 1) ** 3
    # l9_4 = rev_normalize(gaussian_filter(tanh(d17, prop(50)), prop(1))) ** 5 * inland_weight(1)
    # l9 = avgs(l9_1, l9_2, l9_3, l9_4) * l4 * inland_weight(2)  # Top-right

    # l10_1 = sigmoid(gaussian_filter(d22, prop(1)), 1) ** 5
    # l10_2 = rev_normalize(gaussian_filter(tanh(d2, 50), prop(1))) ** 5
    # l10_3 = rev_sigmoid(gaussian_filter(d27, prop(1)), 1) ** 3
    # l10_4 = rev_normalize(gaussian_filter(tanh(d19, 50), prop(1))) ** 5 * inland_weight(1)
    # l10 = avgs(l10_1, l10_2, l10_3, l10_4) * l4 * inland_weight(2)  # Bottom-right
    assert d in {"north", "south"}
    prop = lambda x: x * prop_s.shape[0] // 180
    sig = sigmoid if d == "south" else rev_sigmoid
    rev_sig = rev_sigmoid if d == "south" else sigmoid

    p1 = sig(gaussian_filter(prop_s, 1), 1) ** 5
    p2 = rev_tanh(gaussian_filter(inner_layer(dist_s), 1) / prop(20)) ** 2
    p3 = rev_sig(gaussian_filter(prop_i, 1), 1) ** 3
    p4 = rev_tanh(gaussian_filter(inner_layer(dist_i), 1) / prop(20)) ** 2 * inlandness(1)

    c = avgs(p1, p2, p3, p4) * inlandness(2)
    return c


####################################################################################################
# Process inputs to TemperatureNet inputs
####################################################################################################
def process_inputs(inputs: np.ndarray, resolution: tuple[int, int] = (360, 180)) -> np.ndarray:
    """Load inputs, convert them into useful inputs for TemperatureNet, and returns them.

    :param inputs: A numpy array with 180 * 360 rows and 64 columns
    :param resolution: The resolution of the data

    Requires inputs to be pre-downloaded.
    """
    w, h = resolution
    # inputs = get_inputs(resolution)
    # targets = get_targets(resolution)

    def extract_row(i: int) -> np.ndarray:
        """Extract a row from the raw inputs."""
        return inputs[:, i].reshape((h, w))

    latitude = extract_row(0)
    elevation = extract_row(1)
    inland = extract_row(2)
    water_influence = extract_row(3)

    elev_diff_3_r, elev_diff_3_ur = extract_row(4), extract_row(5)
    elev_diff_3_u, elev_diff_3_ul = extract_row(6), extract_row(7)
    elev_diff_3_l, elev_diff_3_dl = extract_row(8), extract_row(9)
    elev_diff_3_d, elev_diff_3_dr = extract_row(10), extract_row(11)

    elev_diff_5_r, elev_diff_5_ur = extract_row(12), extract_row(13)
    elev_diff_5_u, elev_diff_5_ul = extract_row(14), extract_row(15)
    elev_diff_5_l, elev_diff_5_dl = extract_row(16), extract_row(17)
    elev_diff_5_d, elev_diff_5_dr = extract_row(18), extract_row(19)

    elev_diff_10_r, elev_diff_10_ur = extract_row(20), extract_row(21)
    elev_diff_10_u, elev_diff_10_ul = extract_row(22), extract_row(23)
    elev_diff_10_l, elev_diff_10_dl = extract_row(24), extract_row(25)
    elev_diff_10_d, elev_diff_10_dr = extract_row(26), extract_row(27)

    elev_diff_15_r, elev_diff_15_ur = extract_row(28), extract_row(29)
    elev_diff_15_u, elev_diff_15_ul = extract_row(30), extract_row(31)
    elev_diff_15_l, elev_diff_15_dl = extract_row(32), extract_row(33)
    elev_diff_15_d, elev_diff_15_dr = extract_row(34), extract_row(35)

    # left:                     d14, d7
    # right:                    d13, d8
    # top-right:                d17, d4
    # top-left:                 d18, d3
    # bottom-right:             d19, d2
    # bottom-left:              d20, d1
    # bottom-left to top-right: d21, d28
    # bottom-right to top-left: d22, d27
    # left to right:            d24, d25
    d1, d2, d3, d4 = extract_row(36), extract_row(37), extract_row(38), extract_row(39)
    d5, d6, d7, d8 = extract_row(40), extract_row(41), extract_row(42), extract_row(43)
    d9, d10, d11, d12 = extract_row(44), extract_row(45), extract_row(46), extract_row(47)
    d13, d14, d15, d16 = extract_row(48), extract_row(49), extract_row(50), extract_row(51)
    d17, d18, d19, d20 = extract_row(52), extract_row(53), extract_row(54), extract_row(55)
    d21, d22, d23, d24 = extract_row(56), extract_row(57), extract_row(58), extract_row(59)
    d25, d26, d27, d28 = extract_row(60), extract_row(61), extract_row(62), extract_row(63)

    prop = lambda x: x * h // 180
    elev = lambda x: x / 1000

    # Latitude
    # north_offset = 0  # -90 to 90
    # north_grad = 0  # -1 to 1
    # spread_1 = 1 / 2  # fraction, smaller means more spread
    # l1a = normalize(45 - abs(latitude - north_offset) + north_grad * latitude)
    # l1b = spread(l1a, spread_1)
    l1a = normalize(45 - abs(latitude))
    l1b = spread(l1a, 0.5)
    l1c = normalize(45 - abs(latitude - 15))
    l1d = spread(l1c, 0.5)
    l1e = normalize(45 - abs(latitude + 15))
    l1f = spread(l1e, 0.5)
    l1g = normalize(45 - abs(latitude - 30))
    l1h = spread(l1g, 0.5)
    l1i = normalize(45 - abs(latitude + 30))
    l1j = spread(l1i, 0.5)

    # Elevation, Inland, Water-Influence
    l2 = elev(elevation)
    l3 = tanh(inland / prop(30))
    l4 = tanh(water_influence / prop(20))

    # Coasts
    def cell(start: float, end: float) -> np.ndarray:
        """Return a masking layer of values with latitude boundaries [start, end].
        Values near the boundaries fade out."""
        return relu(-4 * (abs(latitude) - start) * (abs(latitude) - end) / ((start - end) ** 2))
    equator = cell(-30, 30)
    hadley = cell(10, 40)
    ferrel = cell(30, 60)
    poles = cell(50, 130)

    inland_weight = lambda n: (1 - l3) ** n

    # West coast
    l5a, l5b = straight_coast(d24, d7, d25, d14, l4, "west")
    l6 = diagonal_coast(d22, d3, d27, d18, inland_weight, "north")
    l7 = diagonal_coast(d21, d1, d28, d20, inland_weight, "south")

    # East coast
    l8a, l8b = straight_coast(d24, d8, d25, d13, l4, "east")
    l9 = diagonal_coast(d21, d4, d28, d17, inland_weight, "north")
    l10 = diagonal_coast(d22, d2, d27, d19, inland_weight, "south")

    # Rain-shadow
    l11a = elev(avgs(relu(elev_diff_3_r), relu(elev_diff_5_r)))
    l12a = elev(avgs(relu(elev_diff_3_l), relu(elev_diff_5_l)))
    l13a = elev(avgs(relu(elev_diff_3_ul), relu(elev_diff_5_ul)))
    l14a = elev(avgs(relu(elev_diff_3_ur), relu(elev_diff_5_ur)))
    l15a = elev(avgs(relu(elev_diff_3_dl), relu(elev_diff_5_dl)))
    l16a = elev(avgs(relu(elev_diff_3_dr), relu(elev_diff_5_dr)))

    l11b = elev(avgs(relu(elev_diff_10_r), relu(elev_diff_15_r)))
    l12b = elev(avgs(relu(elev_diff_10_l), relu(elev_diff_15_l)))
    l13b = elev(avgs(relu(elev_diff_10_ul), relu(elev_diff_15_ul)))
    l14b = elev(avgs(relu(elev_diff_10_ur), relu(elev_diff_15_ur)))
    l15b = elev(avgs(relu(elev_diff_10_dl), relu(elev_diff_15_dl)))
    l16b = elev(avgs(relu(elev_diff_10_dr), relu(elev_diff_15_dr)))

    # l17a, l18a, l19a = rev_relu(elev_diff_3_r), rev_relu(elev_diff_3_l), rev_relu(elev_diff_3_ul)
    # l20a, l21a, l22a = rev_relu(elev_diff_3_ur), rev_relu(elev_diff_3_dl), rev_relu(elev_diff_3_dr)
    #
    # l17b = avgs(rev_relu(elev_diff_5_r), rev_relu(elev_diff_10_r), rev_relu(elev_diff_15_r))
    # l18b = avgs(rev_relu(elev_diff_5_l), rev_relu(elev_diff_10_l), rev_relu(elev_diff_15_l))
    # l19b = avgs(rev_relu(elev_diff_5_ul), rev_relu(elev_diff_10_ul), rev_relu(elev_diff_15_ul))
    # l20b = avgs(rev_relu(elev_diff_5_ur), rev_relu(elev_diff_10_ur), rev_relu(elev_diff_15_ur))
    # l21b = avgs(rev_relu(elev_diff_5_dl), rev_relu(elev_diff_10_dl), rev_relu(elev_diff_15_dl))
    # l22b = avgs(rev_relu(elev_diff_5_dr), rev_relu(elev_diff_10_dr), rev_relu(elev_diff_15_dr))

    r = lambda x: x.reshape(w * h, 1)
    eq = lambda x: x * equator
    ha = lambda x: x * hadley
    fe = lambda x: x * ferrel
    po = lambda x: x * poles
    n = lambda x: x * (latitude >= 0)
    s = lambda x: x * (latitude <= 0)
    temp_inputs = np.c_[
        r(l1a), r(l1b), r(l1c), r(l1d), r(l1e),                             # latitude
        r(l1f), r(l1g), r(l1h), r(l1i), r(l1j),
        r(eq(l2)), r(ha(l2)), r(fe(l2)), r(po(l2)),                         # elevation
        r(n(eq(l3))), r(n(ha(l3))), r(n(fe(l3))), r(n(po(l3))),             # inland
        r(s(eq(l3))), r(s(ha(l3))), r(s(fe(l3))), r(n(po(l3))),
        r(n(eq(l4))), r(n(ha(l4))), r(n(fe(l4))), r(n(po(l4))),             # water influence
        r(s(eq(l4))), r(s(ha(l4))), r(s(fe(l4))), r(s(po(l4))),
        r(n(eq(l5a))), r(n(ha(l5a))), r(n(fe(l5a))),                        # west coast
        r(s(eq(l5a))), r(s(ha(l5a))), r(s(fe(l5a))),
        r(n(eq(l5b))), r(n(ha(l5b))), r(n(fe(l5b))),
        r(s(eq(l5b))), r(s(ha(l5b))), r(s(fe(l5b))),
        r(n(eq(l6))), r(n(ha(l6))), r(n(fe(l6))),
        r(s(eq(l6))), r(s(ha(l6))), r(s(fe(l6))),
        r(n(eq(l7))), r(n(ha(l7))), r(n(fe(l7))),
        r(s(eq(l7))), r(s(ha(l7))), r(s(fe(l7))),
        r(n(eq(l8a))), r(n(ha(l8a))), r(n(fe(l8a))),                        # east coast
        r(s(eq(l8a))), r(s(ha(l8a))), r(s(fe(l8a))),
        r(n(eq(l8b))), r(n(ha(l8b))), r(n(fe(l8b))),
        r(s(eq(l8b))), r(s(ha(l8b))), r(s(fe(l8b))),
        r(n(eq(l9))), r(n(ha(l9))), r(n(fe(l9))),
        r(s(eq(l9))), r(s(ha(l9))), r(s(fe(l9))),
        r(n(eq(l10))), r(n(ha(l10))), r(n(fe(l10))),
        r(s(eq(l10))), r(s(ha(l10))), r(s(fe(l10))),
        r(n(eq(l11a))), r(n(eq(l11b))), r(n(eq(l16a))), r(n(eq(l16b))),     # rainshadow
        r(s(eq(l11a))), r(s(eq(l11b))), r(s(eq(l14a))), r(s(eq(l14b))),
        r(n(ha(l12a))), r(n(ha(l12b))), r(n(ha(l13a))), r(n(ha(l13b))),
        r(s(ha(l12a))), r(s(ha(l12b))), r(s(ha(l15a))), r(s(ha(l15b))),
        r(n(fe(l12a))), r(n(fe(l12b))), r(n(fe(l13a))), r(n(fe(l13b))),
        r(s(fe(l12a))), r(s(fe(l12b))), r(s(fe(l15a))), r(s(fe(l15b))),
        r(n(po(l11a))), r(n(po(l11b))), r(n(po(l16a))), r(n(po(l16b))),
        r(s(po(l11a))), r(s(po(l11b))), r(s(po(l14a))), r(s(po(l14b)))
    ]
    return temp_inputs


def save_temp_inputs(resolution: tuple[int, int], remove_na: bool = False,
                     flip: tuple[bool, bool, bool] = (False, False, False)) -> None:
    """Download all TemperatureNet inputs in a specified format.

    Requires temp_inputs to be downloaded.

    :param resolution: The resolution of the raw inputs to load
    :param remove_na: Omit na values (ie. remove water)
    :param flip: A tuple of flipping parameters for flipping horizontally, vertically, or to return
    data for non-flipped and all-flipped.
    """
    w, h = resolution
    remove = lambda x: remove_na_rows(x) if remove_na else x
    f1, f2, f3 = flip
    i = remove(process_inputs(get_inputs(resolution, False, (f1, f2, False)), resolution))
    if f3:
        i = np.r_[
            remove(process_inputs(get_inputs(resolution, False, (False, False, False)), resolution)),
            remove(process_inputs(get_inputs(resolution, False, (True, True, False)), resolution))
        ]

    np.save(format_filepath(w, h, remove_na, flip, "temp_inputs"), i)
    print(f"Successfully saved temp_inputs in {w} by {h} resolution, "
          f"remove_na={remove_na}, flip={flip}")


def get_temp_inputs(resolution: tuple[int, int], remove_na: bool = False,
                    flip: tuple[bool, bool, bool] = (False, False, False)) -> np.ndarray:
    """Retrieve the saved temp_inputs."""
    w, h = resolution
    return np.load(format_filepath(w, h, remove_na, flip, "temp_inputs"))
