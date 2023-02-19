"""Climate predictor - the data exploration part

"""
from process_data import *


if __name__ == "__main__":
    resolution = (360, 180)
    w, h = resolution
    inputs = get_inputs(resolution)
    targets = get_targets(resolution)
    targets_flipped = get_targets(resolution, False, (True, True, False))

    extract_row = lambda i: inputs[:, i].reshape((h, w))
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

    # Helper functions
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
    cell = lambda start, end: relu(
        -4 * (abs(latitude) - start) * (abs(latitude) - end) / ((start - end) ** 2)
    )
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
    l11a, l12a, l13a = elev(relu(elev_diff_3_r)), elev(relu(elev_diff_3_l)), elev(
        relu(elev_diff_3_ul))
    l14a, l15a, l16a = elev(relu(elev_diff_3_ur)), elev(relu(elev_diff_3_dl)), elev(
        relu(elev_diff_3_dr))

    l11b = elev(avgs(relu(elev_diff_5_r), relu(elev_diff_10_r), relu(elev_diff_15_r)))
    l12b = elev(avgs(relu(elev_diff_5_l), relu(elev_diff_10_l), relu(elev_diff_15_l)))
    l13b = elev(avgs(relu(elev_diff_5_ul), relu(elev_diff_10_ul), relu(elev_diff_15_ul)))
    l14b = elev(avgs(relu(elev_diff_5_ur), relu(elev_diff_10_ur), relu(elev_diff_15_ur)))
    l15b = elev(avgs(relu(elev_diff_5_dl), relu(elev_diff_10_dl), relu(elev_diff_15_dl)))
    l16b = elev(avgs(relu(elev_diff_5_dr), relu(elev_diff_10_dr), relu(elev_diff_15_dr)))

    month = np.repeat(np.arange(0., 12.).reshape(12, 1), w * h, axis=1)
    month = np.r_[month, month]
    for i in range(24):
        month[i] = month[i] + np.random.randn(w * h) / 4
    data = np.concatenate((targets[0].copy(), targets_flipped[0].copy()), axis=0)
    x = latitude.reshape(w * h)
    y = 70 * np.cos(x * np.pi / 180) - 40 - 4 / (np.e ** (x ** 2 / 400))
    for i in range(24):
        data[i] -= y
    x = elevation.reshape(w * h)
    y = -19 * x / 3000 + 3
    for i in range(24):
        data[i] -= y
