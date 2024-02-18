import glob
import math
import os
import re
from pathlib import Path
from itertools import product

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.sparse.linalg import eigsh
import torch.nn.functional as F
import torch.nn as nn
from typing import Tuple

from math import radians, cos, sin, asin, sqrt, floor


def fit_delimiter(string='', length=80, delimiter="="):
    result_len = length - len(string)
    half_len = math.floor(result_len / 2)
    result = delimiter * half_len + string + delimiter * half_len
    return result


def init_torch_seeds(seed=0):
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def zipdir(path, ziph, include_format):
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] in include_format:
                filename = os.path.join(root, file)
                arcname = os.path.relpath(os.path.join(root, file), os.path.join(path, '..'))
                ziph.write(filename, arcname)


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def get_normalized_features(X):
    # X.shape=(num_nodes, num_features)
    means = np.mean(X, axis=0)  # mean of features, shape:(num_features,)
    X = X - means.reshape((1, -1))
    stds = np.std(X, axis=0)  # std of features, shape:(num_features,)
    X = X / stds.reshape((1, -1))
    return X, means, stds


def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]

    # row sum
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # column sum
    # deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
    deg_mat = deg_mat_row

    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))

    if mat_type == 'com_lap_mat':
        # Combinatorial
        com_lap_mat = deg_mat - adj_mat
        return com_lap_mat
    elif mat_type == 'wid_rw_normd_lap_mat':
        # For ChebConv
        rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat)
        rw_normd_lap_mat = id_mat - rw_lap_mat
        lambda_max_rw = eigsh(rw_lap_mat, k=1, which='LM', return_eigenvectors=False)[0]
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat
        return wid_rw_normd_lap_mat
    elif mat_type == 'hat_rw_normd_lap_mat':
        # For GCNConv
        wid_deg_mat = deg_mat + id_mat
        wid_adj_mat = adj_mat + id_mat
        hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f'ERROR: {mat_type} is unknown.')


def maksed_mse_loss(input, target, mask_value=-1):
    mask = target == mask_value
    out = (input[~mask] - target[~mask]) ** 2
    loss = out.mean()
    return loss

def rotate_loss(head,relation,tail,model):
    out = model.RotatE(head,relation,tail,' ')
    out = F.logsigmoid(out)
    
    loss = -out.mean()
    return loss

def rotate(head,relation,hidden,device):
    pi = 3.14159265358979323846
        
    re_head, im_head = torch.chunk(head, 2, dim=1)

    #Make phases of relations uniformly distributed in [-pi, pi]
    embedding_range = nn.Parameter(
                    torch.Tensor([(24.0 + 2.0) / hidden]), 
                    requires_grad=False
            ).to(device)

    phase_relation = relation/(embedding_range/pi)

    re_relation = torch.cos(phase_relation)
    im_relation = torch.sin(phase_relation)


    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation

    score = torch.cat([re_score, im_score], dim = 1)
    return score

def rotate_batch(head,relation,hidden,device):
    pi = 3.14159265358979323846
        
    re_head, im_head = torch.chunk(head, 2, dim=2)

    #Make phases of relations uniformly distributed in [-pi, pi]
    embedding_range = nn.Parameter(
                    torch.Tensor([(24.0 + 2.0) / hidden]), 
                    requires_grad=False
            ).to(device)

    phase_relation = relation/(embedding_range/pi)

    re_relation = torch.cos(phase_relation)
    im_relation = torch.sin(phase_relation)


    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation

    score = torch.cat([re_score, im_score], dim = 2)
    return score

def spatial_interval(sample):
    s_max = 10
    locs = [each[3] for each in sample[1]]
    num = len(locs)
    s_interval = torch.zeros(num,num)
    for i in range(num):
        for j in range(num):
            s_interval[i,j] = min(haversine(locs[i],locs[j]),s_max)
    return s_interval

def temporal_interval(sample):
    t_max = 10
    timeoffsets = [each[4] for each in sample[1]]
    num = len(timeoffsets)
    t_interval = torch.zeros(num,num)
    for i in range(num):
        for j in range(num):
            t_interval[i,j] = min(abs(timeoffsets[i] - timeoffsets[j])/3600.0,t_max)
    return t_interval

def get_pos(src,sample,dim,mode):
    timeoffsets = [each[4] for each in sample[1]]
    num = len(timeoffsets)
    # timeoffsets = torch.tensor(timeoffsets,dtype=float)
    # af_timeoffsets = timeoffsets[1:]
    # be_timeoffsets = timeoffsets[:-1]
    # offset = (af_timeoffsets - be_timeoffsets) / 3600
    offset = [timeoffsets[i] - timeoffsets[0] for i in range(num)]
    offset = torch.tensor(offset,dtype=float)
    offset = offset / 3600.0
    
    pos = torch.zeros(num)
    pos[0] = 1.0
    
    for i in range(1,num):
        pos[i] = pos[i-1] + offset[i-1]/(sum(offset[:i])/i + 1e-6) + 1.0
        #pos[i] = pos[i-1] + offset[i]
        
    if mode == 'origin':
        pos = torch.arange(num)
        
    pos = pos.unsqueeze(1)
    
    pe = torch.zeros(num, dim)
    # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
    div_term = torch.exp(
        torch.arange(0, dim, 2) * -(math.log(10000.0) / dim)
    )
    # 计算PE(pos, 2i)
    pe[:, 0::2] = torch.sin(pos * div_term)
    # 计算PE(pos, 2i+1)
    pe[:, 1::2] = torch.cos(pos * div_term)
    
    return src + pe.requires_grad_(False)

# 生成旋转矩阵
def precompute_freqs_cis(dim: int, pos: torch.Tensor, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(pos, freqs).float()  # 计算m * \theta

    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    return freqs_cis

# 旋转位置编码计算
def apply_rotary_emb(
    xq: torch.Tensor,
    dim: int,
    sample,
    mode
) -> torch.Tensor:
    
    timeoffsets = [each[4] for each in sample[1]]
    num = len(timeoffsets)
    offset = [timeoffsets[i] - timeoffsets[0] for i in range(num)]
    offset = torch.tensor(offset,dtype=float)
    offset = offset / 3600.0
    
    pos = torch.zeros(num)
    pos[0] = 1.0
    
    for i in range(1,num):
        pos[i] = pos[i-1] + offset[i-1]/(sum(offset[:i])/i + 1e-6) + 1.0
        #pos[i] = pos[i-1] + offset[i]
        
    if mode=='origin':
        pos = torch.arange(num)
    
    freqs_cis = precompute_freqs_cis(dim,pos).to(xq.device)
    
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    
    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    
    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq)
    
def get_sim_of_target(sample,gamma2,alpha2):
    traj_times = [each[4] for each in sample[1]]
    target_times = [each[4] for each in sample[2]]
    num = len(traj_times)
    
    sim = torch.zeros(num,num)
    for i in range(num):
        for j in range(i+1):
            t = abs(traj_times[j] - target_times[i]) / 3600
            sim[i,j] = ((1+cos(gamma2*t))/2)*math.exp(-alpha2*t) + 1e-10
    mask = sim==0.0
    sim[mask] = -1e10
    return F.softmax(sim,dim=-1)
    
def get_sim_of_traj(sample):
    s_interval = spatial_interval(sample)
    t_interval = temporal_interval(sample)
    st_interval = t_interval + s_interval
    st_max = torch.max(st_interval)
    st_interval = st_max - st_interval
    
    num = len(s_interval)
    mask = (torch.triu(torch.ones(num, num))==1).transpose(0,1)
    mask = ~mask
    st_interval[mask] = -1e10

    return F.softmax(st_interval,dim=-1)

def haversine(point_1, point_2):
    lat_1, lng_1 = point_1
    lat_2, lng_2 = point_2
    lat_1, lng_1, lat_2, lng_2 = map(radians, [lat_1, lng_1, lat_2, lng_2])

    d_lon = lng_2 - lng_1
    d_lat = lat_2 - lat_1
    a = sin(d_lat / 2) ** 2 + cos(lat_1) * cos(lat_2) * sin(d_lon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    # return floor(c * r)
    return c * r

def top_k_acc(y_true_seq, y_pred_seq, k):
    hit = 0
    # Convert to binary relevance (nonzero is relevant).
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        top_k_rec = y_pred.argsort()[-k:][::-1]
        idx = np.where(top_k_rec == y_true)[0]
        if len(idx) != 0:
            hit += 1
    return hit / len(y_true_seq)


def mAP_metric(y_true_seq, y_pred_seq, k):
    # AP: area under PR curve
    # But in next POI rec, the number of positive sample is always 1. Precision is not well defined.
    # Take def of mAP from Personalized Long- and Short-term Preference Learning for Next POI Recommendation
    rlt = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        rec_list = y_pred.argsort()[-k:][::-1]
        r_idx = np.where(rec_list == y_true)[0]
        if len(r_idx) != 0:
            rlt += 1 / (r_idx[0] + 1)
    return rlt / len(y_true_seq)


def MRR_metric(y_true_seq, y_pred_seq):
    """Mean Reciprocal Rank: Reciprocal of the rank of the first relevant item """
    rlt = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        rec_list = y_pred.argsort()[-len(y_pred):][::-1]
        r_idx = np.where(rec_list == y_true)[0][0]
        rlt += 1 / (r_idx + 1)
    return rlt / len(y_true_seq)


def top_k_acc_last_timestep(y_true_seq, y_pred_seq, k):
    """ next poi metrics """
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    top_k_rec = y_pred.argsort()[-k:][::-1]
    idx = np.where(top_k_rec == y_true)[0]
    if len(idx) != 0:
        return 1
    else:
        return 0

def top_k_acc(y_true_seq, y_pred_seq, k, poi_dict):
    """ next poi metrics """
    y_true = y_true_seq[-1].item()
    y_pred = y_pred_seq[-1]
    top_k_rec = torch.argsort(y_pred,descending=True).tolist()
    top_k_rec = [poi_dict[-1][each] for each in top_k_rec]
    top_k_rec = top_k_rec[:k]
    if y_true in top_k_rec:
        return 1
    else:
        return 0
    
def cal_MRR(y_true_seq, y_pred_seq,poi_dict):
    y_true = y_true_seq[-1].item()
    y_pred = y_pred_seq[-1]
    rec_list = torch.argsort(y_pred,descending=True).tolist()
    rec_list = [poi_dict[-1][each] for each in rec_list]
    
    if y_true in rec_list:
        r_idx = rec_list.index(y_true) 
    else:
        r_idx = len(rec_list)

    return 1 / (r_idx + 1)
    


def mAP_metric_last_timestep(y_true_seq, y_pred_seq, k):
    """ next poi metrics """
    # AP: area under PR curve
    # But in next POI rec, the number of positive sample is always 1. Precision is not well defined.
    # Take def of mAP from Personalized Long- and Short-term Preference Learning for Next POI Recommendation
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-k:][::-1]
    r_idx = np.where(rec_list == y_true)[0]
    if len(r_idx) != 0:
        return 1 / (r_idx[0] + 1)
    else:
        return 0


def MRR_metric_last_timestep(y_true_seq, y_pred_seq):
    """ next poi metrics """
    # Mean Reciprocal Rank: Reciprocal of the rank of the first relevant item
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-len(y_pred):][::-1]
    r_idx = np.where(rec_list == y_true)[0][0]
    return 1 / (r_idx + 1)


def array_round(x, k=4):
    # For a list of float values, keep k decimals of each element
    return list(np.around(np.array(x), k))

def get_time_slot_id(time):
    minute = time.minute
    hour = time.hour
    day_number = time.dayofweek
    
    if minute <= 30:
        ans = 2*hour
    else:
        ans = 2*hour + 1
    
    if day_number >= 5 :
        return ans+48
    else:
        return ans
    
def get_cat_norm_time12(time):
    hour = (time.hour - 1)
    day_number = time.dayofweek
    ans = hour // 4
    
    if day_number >= 5 :
        ans = ans + 6
    return ans / 12
    
def get_cat_norm_time24(time):
    hour = (time.hour - 1)
    day_number = time.dayofweek
    ans = hour // 2
    
    if day_number >= 5 :
        ans = ans + 12
    return ans / 24

def get_cat_norm_time48(time):
    hour = (time.hour - 1)
    day_number = time.dayofweek
    ans = hour
    
    if day_number >= 5 :
        ans = ans + 24
    return ans / 48
    
def get_norm_time_id24(hour,min):
    return hour

def get_norm_time24(time):
    min = time.minute
    hour = time.hour
    
    return get_norm_time_id24(hour,min) / 24

def get_norm_time_id48(hour,min):
    return 2 * hour if min < 30 else 2 * hour + 1

def get_norm_time12(time):
    hour = time.hour
    hour = hour // 4
    day_number = time.dayofweek
    if day_number >= 5:
        hour += 6
    return hour / 12

def get_norm_time24(time):
    hour = time.hour
    day_number = time.dayofweek
    
    hour = hour // 2
    if day_number >= 5:
        hour += 12
    
    return hour / 24
    
def get_norm_time48(time):
    hour = time.hour
    day_number = time.dayofweek
    
    if day_number >= 5:
        hour += 24
    
    return hour/ 48

def get_norm_time96(time):
    hour = time.hour
    minute = time.minute
    
    ans = minute//15 + 4*hour
    
    return ans

def get_norm_time192(time):
    hour = time.hour
    minute = time.minute
    day_number = time.dayofweek
    
    ans = minute//15 + 4*hour
    
    if day_number >= 5:
        ans += 96
    
    return ans/192
    

def get_norm_week(time):
    week = time.week
    week_of_month = week % 5
    return week_of_month / 5

def get_user_popular(df,poi_id2idx_dict,user_id2idx_dict):
    user_popular_dict = dict()
    grouped = df.groupby('user_id')
    for name,value in grouped:
        top = list(value['POI_id'].value_counts().to_dict().keys())
        #top = list(set(value['POI_id'].to_list()))
        top = [poi_id2idx_dict[each] for each in top]
        user_popular_dict[user_id2idx_dict[name]] = top
    return user_popular_dict

def get_user_popular_and_fre(df,poi_id2idx_dict,user_id2idx_dict):
    user_popular_dict = dict()
    user_popular_fre_dict = dict()
    grouped = df.groupby('user_id')
    for name,value in grouped:
        poi_fre = value['POI_id'].value_counts().to_dict()
        top = list(value['POI_id'].value_counts().to_dict().keys())
        fre = [poi_fre[each] for each in top]
        
        top = [poi_id2idx_dict[each] for each in top]
        fre_sum = sum(fre)
        fre = [each / fre_sum for each in fre]
        
        user_popular_dict[user_id2idx_dict[name]] = top
        user_popular_fre_dict[user_id2idx_dict[name]] = fre
    return user_popular_dict,user_popular_fre_dict

def get_all_permutations_dict(length):
    characters = ['0', '1', '2', '3']

    # 生成所有可能的长度为6的字符串
    all_permutations = [''.join(p) for p in product(characters, repeat=length)]

    premutation_dict = dict(zip(all_permutations,range(len(all_permutations))))

    return premutation_dict

def get_day_norm7(time):
    day_number = time.dayofweek
    return day_number

def get_day_norm2(time):
    day_number = time.dayofweek
    if day_number >= 5:
        index = 1
    else:
        index = 0
    return index

def get_month_norm4(time):
    month = time.month
    if month == 12:
        index = 0
    else:
        index = (month // 3)
    
    return index / 4

def get_month_norm12(time):
    month = time.month
    return month / 12

def crotate_dist(x1,x2,mode):
    if mode == 'can':
        scores = []
        batch_size = x1.size(0)
        poi_num = x2.size(0)
        for i in range(batch_size):
            x1_ = x1[i]
            x1_ = x1_.unsqueeze(1)
            x1_ = x1_.expand(-1,poi_num,-1)
            
            x1_re,x1_im = torch.chunk(x1_,2,dim=-1)
            x2_re,x2_im = torch.chunk(x2,2,dim=-1)

            re = x1_re - x2_re
            im = x1_im - x2_im

            score = torch.stack([re,im],dim=0)
            score = score.norm(dim=0)
            score = torch.sum(score,dim=-1)
            score = score.neg()
            
            scores.append(score)
        return torch.stack(scores)
    elif mode == 'neg':
        neg_num = x2.size(2)
        x1 = x1.unsqueeze(2)
        # b s neg_num d
        x1 = x1.expand(-1,-1,neg_num,-1)
        
    x1_re,x1_im = torch.chunk(x1,2,dim=-1)
    x2_re,x2_im = torch.chunk(x2,2,dim=-1)
    
    re = x1_re - x2_re
    im = x1_im - x2_im
    
    score = torch.stack([re,im],dim=0)
    score = score.norm(dim=0)
    score = torch.sum(score,dim=-1)
    score = score.neg()
    
    return score
    
        
        
        

