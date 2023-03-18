import numpy as np
from scipy.stats import logistic, johnsonsb, t, burr


def calc_same_group_probability(x):
    # OUMVLP mixed
    # args = {'loc': 0.27859749153604235, 'scale': 0.05633656136685185}
    # res = logistic.cdf(x=x, **args)

    # HID-500 rerank false
    # args = {'df': 3.2361253485923935, 'loc': 6.4562757079881, 'scale': 0.3990502955897156}
    # res = t.cdf(x=x, **args)

    # HID-500 rerank true
    # args = {'c': 5.784486998578288, 'd': 0.3037424629072403, 'loc': 0.07058832791006939, 'scale': 0.4148182819951845}
    # res = burr.cdf(x=x, **args)
    
    # HID2-500 rerank false
    args = {'df': 3.0381996114844823, 'loc': 5.780107752239095, 'scale': 0.3561692739542963}
    res = t.cdf(x=x, **args)
    
    # HID2-500 rerank true
    # args = {'c': 5.971231187983179, 'd': 0.2623746981586892, 'loc': 0.08102473412269318, 'scale': 0.42178200748044614}
    # res = burr.cdf(x=x, **args)

    return res


def calc_diff_group_probability(x):
    # OUMVLP mixed
    # args = {'a': -0.9611034224157146, 'b': 1.8999539435483852, 'loc': 0.0004702090039294164, 'scale': 1.5349745303565898}
    # res = johnsonsb.cdf(x=x, **args)

    # HID-500 rerank false
    # args = {'a': -7.577993663612768, 'b': 11.961683606640753, 'loc': -25.94841671081837, 'scale': 55.437723214890624}
    # res = johnsonsb.cdf(x=x, **args)

    # HID-500 rerank true
    # args = {'a': -6.9935866012846475, 'b': 7.069360686837677, 'loc': 0.27337035792812575, 'scale': 0.8924954641115803}
    # res = johnsonsb.cdf(x=x, **args)
    
    # HID2-500 rerank false
       
    args = {'a': -6.5782410560868385, 'b': 95.75613055419649, 'loc': -148.47793130445035, 'scale': 304.2028069752622}
    res = johnsonsb.cdf(x=x, **args)

    # HID2-500 rerank true
    # args = {'a': -38.88485304054575, 'b': 89.71112657288847, 'loc': -4.494130773537172, 'scale': 8.913610435700384}
    # res = johnsonsb.cdf(x=x, **args)

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
    w1, w2 = 1.0, 0
    dict = {}
    for dist in np.arange(0., 1.5, 0.01):
        res = calc_similarity(dist, w1, w2)
        dict[dist] = res
    print(dict)
