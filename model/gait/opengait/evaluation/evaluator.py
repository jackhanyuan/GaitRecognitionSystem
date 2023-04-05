import torch
import numpy as np
import torch.nn.functional as F
from opengait_main import opt
from tools import get_msg_mgr, mkdir, config_loader
from .calculate_probability import calc_similarity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cuda_dist(x, y, metric='euc'):
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    if metric == 'cos':
        x = F.normalize(x, p=2, dim=2)  # n p c
        y = F.normalize(y, p=2, dim=2)  # n p c
    num_bin = x.size(1)
    n_x = x.size(0)
    n_y = y.size(0)
    dist = torch.zeros(n_x, n_y).to(device)
    for i in range(num_bin):
        _x = x[:, i, ...]
        _y = y[:, i, ...]
        if metric == 'cos':
            dist += torch.matmul(_x, _y.transpose(0, 1))
        else:
            _dist = torch.sum(_x ** 2, 1).unsqueeze(1) + torch.sum(_y ** 2, 1).unsqueeze(
                1).transpose(0, 1) - 2 * torch.matmul(_x, _y.transpose(0, 1))
            dist += torch.sqrt(F.relu(_dist))
    return 1 - dist / num_bin if metric == 'cos' else dist / num_bin


def re_ranking(original_dist, query_num, k1, k2, lambda_value):
    # Modified from https://github.com/michuanhaohao/reid-strong-baseline/blob/master/utils/re_ranking.py
    all_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    # print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(
                np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                                            :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(
                candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + \
        original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist
    

def evaluate_similarity(data, dataset, metric='euc'):
    cfgs = config_loader(opt.cfgs)
    re_rank = cfgs['evaluator_cfg']['re_rank']
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
