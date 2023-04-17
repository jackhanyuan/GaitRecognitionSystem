import numpy as np
from scipy.stats import logistic, johnsonsb, t, burr


def calc_same_group_probability(x):
    # HID500-OutdoorGait138
    args = {'df': 4.99535496370145, 'loc': 9.62629510552382, 'scale': 0.915049924855037}
    res = t.cdf(x=x, **args)

    return res


def calc_diff_group_probability(x):
    # HID500-OutdoorGait138
    args = {'a': 22.195523809517333, 'b': 27.46516021982316, 'loc': -48.1526371634257, 'scale': 205.14459476282795}
    res = johnsonsb.cdf(x=x, **args)

    return res


def calc_similarity(dist, w1=1.0, w2=0):
    if abs(w1 + w2 - 1.0) > 0.01:
        print("Error! The sum of w1 and w2 must be 1.0")
        return None
    a = calc_same_group_probability(x=dist)
    b = calc_diff_group_probability(x=dist)
    similarity = w1 * (1-a) + w2 * (1-b)
    return similarity


if __name__ == '__main__':
    w1, w2 = 1.0, 0.0
    dict = {}
    for dist in np.arange(0., 26.0, 0.01):
        res = calc_similarity(dist, w1, w2)
        dict[dist] = res
    print(dict)
