import numpy as np
from opengait_main import opt
from tools import get_msg_mgr, mkdir, config_loader
from .calculate_probability import calc_similarity
from .re_rank import re_ranking
from .metric import cuda_dist


def evaluate_similarity(data, dataset, metric='euc'):
    cfgs = config_loader(opt.cfgs)
    re_rank = cfgs['evaluator_cfg']['rerank']
    rank_k = 6

    msg_mgr = get_msg_mgr()
    msg_mgr.log_info("Evaluating Dist")
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    label = np.array(label)
    seq_type = np.array(seq_type)

    gallery_mask = (label != "probe")
    probe_mask = (label == "probe")

    gallery_feature = feature[gallery_mask, :]
    gallery_label = label[gallery_mask]
    gallery_seq_type = seq_type[gallery_mask]

    probe_feature = feature[probe_mask, :]
    probe_label = seq_type[probe_mask]

    if re_rank:
        print('starting re_ranking')
        feat = np.concatenate([probe_feature, gallery_feature])
        dist = cuda_dist(feat, feat, metric).cpu().numpy()
        dist = re_ranking(dist, probe_feature.shape[0], k1=6, k2=6, lambda_value=0.3)
    else:
        dist = cuda_dist(probe_feature, gallery_feature, metric).cpu().numpy()

    idx = np.argsort(dist, axis=1)
    simi = list(map(calc_similarity, dist))
    res = {}
    n = min(len(idx[0]), rank_k)  # 只要排名前rank_k的
    for i in range(len(idx)):
        label = {}
        for j in range(n):
            index = idx[i, j]
            g_label = str(gallery_label[index] + "-" + gallery_seq_type[index])
            label[g_label] = {"dist": float(dist[i][index]), "similarity": float(simi[i][index])}
        res[probe_label[i]] = label

    msg_mgr.log_info(res)

    # for i in res:
    #     print(i)
    #     for j in res[i]:
    #         print("\t{0:20}\t{1:5.3f}\t{2:6.3f}%".format(j, res[i][j]["dist"], res[i][j]["similarity"] * 100))

    return res
