
import torch
import random
import numpy as np

def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    print('set seeds over!!')
set_seeds(3407)

import logging
import logging
import os
import pathlib
import pickle
import zipfile
from pathlib import Path
import math
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from timm.scheduler.cosine_lr import CosineLRScheduler
import statistics


from old_model import  UserEmbeddings, GPSEmbeddings,OriginTime2Vec, GPSEncoder, TransformerModel,PoiEmbeddings,TimeEncoder,CatTime2Vec,FuseEmbeddings,OriginUserTime2Vec,TimeEmbeddings,MLP
from param_parser import parameter_parser
from utils import increment_path, zipdir, top_k_acc_last_timestep,get_all_permutations_dict, \
    mAP_metric_last_timestep, MRR_metric_last_timestep, maksed_mse_loss,rotate_loss,rotate,get_sim_of_traj,haversine,get_day_norm7
from utils import get_cat_norm_time12,get_cat_norm_time24,get_cat_norm_time48,get_user_popular,get_user_popular_and_fre
from quad_key_encoder import latlng2quadkey
from nltk import ngrams
from utils import get_norm_time12,get_norm_time24,get_norm_time48,get_norm_time96,get_norm_time192

def get_norm_time_id(hour,min):
    return 2 * hour if min < 30 else 2 * hour + 1

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
def get_ngrams_of_quadkey(quadkey,n,permutations_dict):
    region_quadkey_bigram = ' '.join([''.join(x) for x in ngrams(quadkey, n)])
    region_quadkey_bigram = region_quadkey_bigram.split()
    region_quadkey_bigram = [permutations_dict[each] for each in region_quadkey_bigram]
    return region_quadkey_bigram

def train(args):
    args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, sep='-')
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    # Setup logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(args.save_dir, f"log_training.txt"),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.info(f"Modified PyTorch random seed: {torch.initial_seed()}")
    logging.info(f"Modified PyTorch CUDA random seed: {torch.cuda.initial_seed()}")
    logging.info(f"Modified NumPy random seed: {np.random.get_state()[1][0]}")
    logging.info(f"Modified Python random module seed: {random.getstate()[1][0]}")
    
    rand1 = random.random()
    rand2 = random.random()
    rand3 = random.random()
    logging.info(f"rand1 {rand1}")
    logging.info(f"rand2 {rand2}")
    logging.info(f"rand3 {rand3}")
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True

    # Save run settings
    logging.info(args)
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # Save python code
    zipf = zipfile.ZipFile(os.path.join(args.save_dir, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
    zipf.close()
    
    permutations_dict = get_all_permutations_dict(args.ngrams)

    # %% ====================== Load data ======================
    # Read check-in train data
    train_df = pd.read_csv(args.data_train,parse_dates=[args.time_feature])
    val_df = pd.read_csv(args.data_val,parse_dates=[args.time_feature])
    
    poi_np = np.load(args.poi_pre_embedding)
    poi_pre_embeddings = torch.tensor(poi_np,dtype = torch.float,requires_grad=True).to(device=args.device)
    poi_pre_embeddings = nn.Parameter(poi_pre_embeddings,requires_grad=True)
    
    poi_id2idx_dict = {}
    with open(args.poi_entity) as file:
        for line in file:
            pidx,pid = line.strip().split('\t')
            pidx = int(pidx)
            pid = int(pid)
            poi_id2idx_dict[pid] = pidx  
        
    pois = list(set(train_df['POI_id'].tolist()))
    num_pois = len(pois)
    assert len(poi_id2idx_dict) == len(pois)
    logging.info(f'poi num is {num_pois}')
    #poi_id2idx_dict = dict(zip(pois,range(num_pois)))
    
    cats = list(set(train_df['POI_catid'].to_list()))
    num_cats = len(cats)
    cat_id2idx_dict = dict(zip(cats,range(num_cats)))
    
    users = list(set(train_df['user_id'].to_list()))
    num_users = len(users)
    user_id2idx_dict = dict(zip(users,range(num_users)))
    
    user_popular,user_popular_fre = get_user_popular_and_fre(train_df,poi_id2idx_dict,user_id2idx_dict)
    
    # %% ====================== Define Dataset ======================
    class TrajectoryDatasetTrain(Dataset):
        def __init__(self, train_df):
            self.df = train_df
            self.traj_seqs = []  
            self.input_seqs = []
            self.input_next_times = []
            self.label_seqs = []

            for traj_id in tqdm(set(train_df[args.traj_feature].tolist())):
                traj_df = train_df[train_df[args.traj_feature] == traj_id]
                poi_ids = traj_df['POI_id'].to_list()
                poi_idxs = [poi_id2idx_dict[each] for each in poi_ids]
                
                cat_ids = traj_df['POI_catid'].to_list()
                cat_idxs = [cat_id2idx_dict[cat] for cat in cat_ids]
                
                user_id = traj_df.iloc[0]['user_id']
                
                time_feature = traj_df[args.time_feature].to_list()
                # time_feature = [get_time_slot_id(time) for time in time_feature]
                norm_time = [get_norm_time96(time)/96 for time in time_feature]
                day_time = [get_day_norm7(time)/7 for time in time_feature]
                
                time_id = [get_time_slot_id(time) for time in  time_feature]
                
                latitudes = traj_df['latitude'].to_list()
                longitudes = traj_df['longitude'].to_list()
                poi_locs = list(zip(latitudes,longitudes))
                
                quad_keys = [latlng2quadkey(loc[0],loc[1],args.quadkey_len) for loc in poi_locs]
                quad_keys = [get_ngrams_of_quadkey(quad_key,args.ngrams,permutations_dict) for quad_key in quad_keys]
                
                timeoffsets = traj_df['UTCTimeOffsetEpoch'].to_list()
                
                input_seq = []
                input_next_time = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], norm_time[i],cat_idxs[i],poi_locs[i],timeoffsets[i],time_id[i],quad_keys[i],day_time[i]))
                    label_seq.append((poi_idxs[i + 1], norm_time[i + 1],cat_idxs[i+1],poi_locs[i+1],timeoffsets[i+1],time_id[i],quad_keys[i+1],day_time[i+1]))
                    input_next_time.append(norm_time[i+1])

                if len(input_seq) < args.short_traj_thres or len(input_seq) >100:
                    continue

                self.traj_seqs.append([traj_id,user_id])
                self.input_seqs.append(input_seq)
                self.input_next_times.append(input_next_time)
                self.label_seqs.append(label_seq)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs) == len(self.input_next_times)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index],self.input_next_times[index])

    class TrajectoryDatasetVal(Dataset):
        def __init__(self, df):
            self.df = df
            self.traj_seqs = []
            self.input_seqs = []
            self.input_next_times = []
            self.label_seqs = []

            for traj_id in tqdm(set(df[args.traj_feature].tolist())):
                # Ger POIs idx in this trajectory
                traj_df = df[df[args.traj_feature] == traj_id]
                
                user_id = traj_df.iloc[0]['user_id']

                # Ignore user if not in training set
                if user_id not in user_id2idx_dict.keys():
                    continue 
                
                poi_ids = traj_df['POI_id'].to_list()
                cat_ids = traj_df['POI_catid'].to_list()
                time_feature = traj_df[args.time_feature].to_list()
                # time_feature = [get_time_slot_id(time) for time in time_feature]
                norm_time = [get_norm_time96(time)/96 for time in time_feature]
                day_time = [get_day_norm7(time)/7 for time in time_feature]
                
                time_id = [get_time_slot_id(time) for time in  time_feature]
                
                latitudes = traj_df['latitude'].to_list()
                longitudes = traj_df['longitude'].to_list()
                poi_locs = list(zip(latitudes,longitudes))
                
                quad_keys = [latlng2quadkey(loc[0],loc[1],args.quadkey_len) for loc in poi_locs]
                quad_keys = [get_ngrams_of_quadkey(quad_key,args.ngrams,permutations_dict) for quad_key in quad_keys]
                
                poi_timeoffsets = traj_df['UTCTimeOffsetEpoch'].to_list()
                
                poi_idxs = []
                time_idxs = []
                cat_idxs = []
                locs = []
                timeoffsets = []
                time_ids = []
                quads = []
                days = []

                for i in range(len(poi_ids)):
                    if poi_ids[i] in poi_id2idx_dict.keys() and cat_ids[i] in cat_id2idx_dict.keys():
                        poi_idxs.append(poi_id2idx_dict[poi_ids[i]])
                        time_idxs.append(norm_time[i])
                        cat_idxs.append(cat_id2idx_dict[cat_ids[i]])
                        locs.append(poi_locs[i])
                        timeoffsets.append(poi_timeoffsets[i])
                        time_ids.append(time_id[i])
                        quads.append(quad_keys[i])
                        days.append(day_time[i])
                    else:
                        # Ignore poi if not in training set
                        continue

                # Construct input seq and label seq
                input_seq = []
                input_next_time = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_idxs[i],cat_idxs[i],locs[i],timeoffsets[i],time_ids[i],quads[i],days[i]))
                    label_seq.append((poi_idxs[i + 1], time_idxs[i + 1],cat_idxs[i+1],locs[i+1],timeoffsets[i+1],time_ids[i+1],quads[i+1],days[i+1]))
                    input_next_time.append(time_idxs[i+1])

                # Ignore seq if too short
                if len(input_seq) < args.short_traj_thres or len(input_seq) > 100:
                    continue

                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)
                self.input_next_times.append(input_next_time)
                self.traj_seqs.append([traj_id,user_id])

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs) == len(self.input_next_times)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index],self.input_next_times[index])

    # %% ====================== Define dataloader ======================
    print('Prepare dataloader...')
    train_dataset = TrajectoryDatasetTrain(train_df)
    val_dataset = TrajectoryDatasetVal(val_df)
    #user_historys = get_user_popular(train_df,poi_id2idx_dict,user_id2idx_dict)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch,
                              shuffle=True, drop_last=False,
                              pin_memory=True, num_workers=args.workers,
                              collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch,
                            shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=args.workers,
                            collate_fn=lambda x: x)

    # %% ====================== Build Models ======================
    # %% Model1: User embedding model, nn.embedding
    user_embed_model = UserEmbeddings(num_users, args.user_embed_dim)
    poi_embed_model = PoiEmbeddings(num_pois,args.poi_embed_dim)
    
    # %% Model2: Time Model
    #time_embed_model = Time2Vec('sin',out_dim=args.time_embed_dim)
    time_embed_model_user = OriginTime2Vec('sin',int(0.5*(args.user_embed_dim+args.poi_embed_dim)))
    time_embed_model_user_tgt = OriginTime2Vec('sin',int(0.5*(args.user_embed_dim+args.poi_embed_dim)))
    #time_embed_model_user = TimeEmbeddings(96,int(0.5*(args.user_embed_dim+args.poi_embed_dim)))
    time_embed_model_user_day = OriginTime2Vec('sin',int(0.5*(args.user_embed_dim+args.poi_embed_dim)))
    time_embed_model_user_day_tgt = OriginTime2Vec('sin',int(0.5*(args.user_embed_dim+args.poi_embed_dim)))
    #time_embed_model_user_day = TimeEmbeddings(2,int(0.5*(args.user_embed_dim+args.poi_embed_dim)))
    
    """ check_point = torch.load(args.poi_time_embed_state_dict)
    check_point2 = torch.load(args.poi_time_tgt_embed_state_dict) """
    
    time_embed_model_poi = OriginTime2Vec('sin',2*args.time_embed_dim)
    time_embed_model_poi_tgt = OriginTime2Vec('sin',2*args.time_embed_dim)
    
    """ time_embed_model_poi.load_state_dict(check_point['time_embed_model_state_dict'])
    time_embed_model_poi_tgt.load_state_dict(check_point2['time_embed_model_state_dict_tgt']) """
    
    #time_embed_model_poi = TimeEmbeddings(96,args.time_embed_dim)
    time_embed_model_poi_day = OriginTime2Vec('sin',2*args.time_embed_dim)
    time_embed_model_poi_day_tgt = OriginTime2Vec('sin',2*args.time_embed_dim)
    #time_embed_model_poi_day = TimeEmbeddings(2,args.time_embed_dim)
    
    
    # Geography model
    gps_embed_model = GPSEmbeddings(4096,args.gps_embed_dim)
    gps_encoder = GPSEncoder(args.gps_embed_dim,1,2*args.gps_embed_dim,2,0.3)
    # %% Model: Sequence model
    args.seq_input_embed = args.user_embed_dim + args.poi_embed_dim 
    # args.seq_input_embed = args.poi_embed_dim  + args.user_embed_dim + 2*args.time_embed_dim
    # args.seq_input_embed = args.poi_embed_dim
    seq_model = TransformerModel(num_pois,
                                 args.seq_input_embed,
                                 args.transformer_nhead,
                                 args.transformer_nhid,
                                 args.transformer_nlayers,
                                 args.time_embed_dim,
                                 args,
                                 dropout=0.4) 

    # Define overall loss and optimizer
    optimizer = optim.Adam(params=list(user_embed_model.parameters()) +
                                  list(time_embed_model_user.parameters()) +
                                  list(time_embed_model_user_tgt.parameters()) +
                                  list(time_embed_model_user_day.parameters()) +
                                  list(time_embed_model_user_day_tgt.parameters()) +
                                  [poi_pre_embeddings] +
                                  list(time_embed_model_poi.parameters())+
                                  list(time_embed_model_poi_tgt.parameters()) +
                                  list(time_embed_model_poi_day.parameters()) +
                                  list(time_embed_model_poi_day_tgt.parameters()) +
                                  list(gps_embed_model.parameters()) +
                                  list(gps_encoder.parameters()) +
                                  list(seq_model.parameters()),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding

    
    lr_scheduler = CosineLRScheduler(optimizer=optimizer,
                                     t_initial=args.epochs,
                                     lr_min=1e-5,
                                     warmup_t=10,
                                     warmup_lr_init=args.warmup_lr_init) 

    # %% Tool functions for training
    def get_rotation_and_loss(sample,label_pois,label_cats,mode):
        # Parse sample
        traj_id = sample[0][0]
        input_seq = [each[0] for each in sample[1]]
        input_seq = torch.tensor(input_seq,dtype=torch.long).to(args.device)
        seq_len = len(input_seq)
        
        input_next_seq = [each[0] for each in sample[2]]
        input_next_seq = torch.tensor(input_next_seq,dtype=torch.long).to(args.device)
        
        time_seconds = [each[4] for each in sample[1]]
        time_interval = [(time_seconds[i] - time_seconds[i-1])/3600 for i in range(1,len(sample[1]))]
        time_interval.insert(0,0)
        m = statistics.mean(time_interval)
        time_interval = [(each+1e-6) / (7.5+1e-6) for each in time_interval]
        time_interval = torch.tensor(time_interval,dtype=torch.float,device=args.device)
        
        locs = [each[3] for each in sample[1]]
        dist_interval = [haversine(locs[i],locs[i-1]) for i in range(1,len(sample[1]))]
        dist_interval.insert(0,0.0)
        mean_dist = statistics.mean(dist_interval)
        dist_interval = [(each+1e-6) / (4.0 + 1e-6) for each in dist_interval]
        dist_interval = torch.tensor(dist_interval,dtype=torch.float,device=args.device)
        
        input_seq_time = [each[1] for each in sample[1]]
        input_seq_time = torch.tensor(input_seq_time,dtype=torch.float).to(args.device)
        
        input_seq_day_time = [each[7] for each in sample[1]]
        input_seq_day_time = torch.tensor(input_seq_day_time,dtype=torch.float).to(args.device)
        
        input_seq_time_id = [each[5] for each in sample[1]]
        input_seq_time_id = torch.tensor(input_seq_time_id,dtype=torch.long).to(args.device)
        
        input_seq_cat = [each[2] for each in sample[1]]
        input_seq_cat = torch.tensor(input_seq_cat,dtype=torch.long).to(args.device)
        
        input_next_time = [each[1] for each in sample[2]]
        input_next_time = torch.tensor(input_next_time,dtype=torch.float).to(args.device)
        
        input_next_day_time = [each[7] for each in sample[2]]
        input_next_day_time = torch.tensor(input_next_day_time,dtype=torch.float).to(args.device)
        
        input_next_time_id = [each[5] for each in sample[2]]
        input_next_time_id = torch.tensor(input_next_time_id,dtype=torch.long).to(args.device)
        
        input_seq_gps = [each[6] for each in sample[1]]
        input_seq_gps = torch.tensor(input_seq_gps,dtype=torch.long).to(args.device)
        
        input_seq_gps_embeddings = gps_embed_model(input_seq_gps)
        input_seq_gps_embeddings = gps_encoder(input_seq_gps_embeddings)
        
        # User to embedding
        user_id = sample[0][1]
        user_idx = user_id2idx_dict[user_id]
        input = torch.LongTensor([user_idx]).to(device=args.device)
        user_embedding = user_embed_model(input)
        user_embedding = torch.squeeze(user_embedding)
        user_embeddings = user_embedding.repeat(seq_len,1).to(args.device) 
        
        user_times = time_embed_model_user(input_seq_time) 
        user_day_times = time_embed_model_user_day(input_seq_day_time)
        
        #seq_poi_embeddings = poi_embed_model(input_seq)
        seq_poi_embeddings = torch.index_select(poi_pre_embeddings,0,input_seq)
        poi_embeds = seq_poi_embeddings
        
        
        user_next_times = time_embed_model_user_tgt(input_next_time)
        user_next_day_times = time_embed_model_user_day_tgt(input_next_day_time)

        poi_next_times = time_embed_model_poi_tgt(input_next_time)
        poi_next_day_times = time_embed_model_poi_day_tgt(input_next_day_time)
        
        poi_times = time_embed_model_poi(input_seq_time)
        poi_day_times = time_embed_model_poi_day(input_seq_day_time)
        
        user_embeddings = torch.cat((user_embeddings,seq_poi_embeddings),dim=-1)

        user_rotate_hour = rotate(user_embeddings,user_times,int(0.5*(args.user_embed_dim+args.poi_embed_dim)),args.device)
        user_rotate_day = rotate(user_embeddings,user_day_times,int(0.5*(args.user_embed_dim+args.poi_embed_dim)),args.device)

        
        user_rotate = 0.7*user_rotate_hour + 0.3*user_rotate_day 
        
        seq_poi_embeddings = torch.cat((seq_poi_embeddings,input_seq_gps_embeddings),dim=-1)


        poi_rotate_hour = rotate(seq_poi_embeddings,poi_times,2*args.time_embed_dim,args.device)
        poi_rotate_day = rotate(seq_poi_embeddings,poi_day_times,2*args.time_embed_dim,args.device)

        poi_rotate = 0.7*poi_rotate_hour + 0.3*poi_rotate_day
        
        seq_embedding1 = user_rotate
        #seq_embedding1 = torch.cat((user_embeddings,seq_poi_embeddings,input_seq_gps_embeddings,user_times),dim=-1)
        seq_embedding2 = poi_rotate
        seq_embedding3 = torch.cat((user_next_times,poi_next_times),dim=-1)
        seq_embedding4 = torch.cat((user_next_day_times,poi_next_day_times),dim=-1)

        if mode == 'train':
            return seq_embedding1,seq_embedding2,seq_embedding3,seq_embedding4,poi_embeds,input_seq_gps_embeddings
        else:
            return seq_embedding1,seq_embedding2,seq_embedding3,seq_embedding4,poi_embeds,input_seq_gps_embeddings
        
    # %% ====================== Train ======================
    user_embed_model = user_embed_model.to(device=args.device)
    poi_embed_model = poi_embed_model.to(device=args.device)

    time_embed_model_user = time_embed_model_user.to(device=args.device)
    time_embed_model_user_tgt = time_embed_model_user_tgt.to(device=args.device)
    #time_embed_model_cat = time_embed_model_cat.to(device=args.device)
    time_embed_model_poi = time_embed_model_poi.to(device=args.device)
    time_embed_model_poi_tgt = time_embed_model_poi_tgt.to(device=args.device)
    
    time_embed_model_poi_day = time_embed_model_poi_day.to(args.device)
    time_embed_model_poi_day_tgt = time_embed_model_poi_day_tgt.to(args.device)
    
    time_embed_model_user_day = time_embed_model_user_day.to(args.device)
    time_embed_model_user_day_tgt = time_embed_model_user_day_tgt.to(args.device)
    
    seq_model = seq_model.to(device=args.device)

    gps_embed_model = gps_embed_model.to(device=args.device)
    gps_encoder = gps_encoder.to(device=args.device)

    # %% Loop epoch
    # For plotting
    train_epochs_top1_acc_list = []
    train_epochs_top5_acc_list = []
    train_epochs_top10_acc_list = []
    train_epochs_top20_acc_list = []
    train_epochs_mAP20_list = []
    train_epochs_mrr_list = []
    train_epochs_loss_list = []
    train_epochs_poi_loss_list = []
    train_epochs_time_loss_list = []
    train_epochs_cat_loss_list = []
    val_epochs_top1_acc_list = []
    val_epochs_top5_acc_list = []
    val_epochs_top10_acc_list = []
    val_epochs_top20_acc_list = []
    val_epochs_mAP20_list = []
    val_epochs_mrr_list = []
    val_epochs_loss_list = []
    val_epochs_poi_loss_list = []
    val_epochs_time_loss_list = []
    val_epochs_cat_loss_list = []
    # For saving ckpt
    max_val_score = -np.inf
    best_score = 0.0
    best_score_list = []

    for epoch in range(args.epochs):
        logging.info(f"{'*' * 50}Epoch:{epoch:03d}{'*' * 50}\n")

        user_embed_model.train()
        poi_embed_model.train()
        
        time_embed_model_user.train()
        time_embed_model_user_tgt.train()
        
        time_embed_model_poi.train()
        time_embed_model_poi_tgt.train()
        
        time_embed_model_poi_day.train()
        time_embed_model_poi_day_tgt.train()
        
        time_embed_model_user_day.train()
        time_embed_model_user_day_tgt.train()
        
        seq_model.train()

        gps_encoder.train()
        gps_embed_model.train()
        


        train_batches_top1_acc_list = []
        train_batches_top5_acc_list = []
        train_batches_top10_acc_list = []
        train_batches_top20_acc_list = []
        train_batches_mAP20_list = []
        train_batches_mrr_list = []
        train_batches_loss_list = []
        train_batches_poi_loss_list = []
        train_batches_time_loss_list = []
        train_batches_cat_loss_list = []
        # Loop batch
        for b_idx, batch in enumerate(train_loader):
            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embed1s = []
            batch_seq_embed2s = []
            batch_seq_labels_poi = []
 
            target_times = []
            target_days = []
            gps_embeds = []
            poi_embeds = []
            
            # Convert input seq to embeddings
            #loss_rotate = 0.0
            import time
            
            for sample in batch:
                
                # sample[0]: traj_id, sample[1]: input_seq_features, sample[2]: label_features
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                label_seq_time = [each[1] for each in sample[2]]
                label_seq_cats = [each[2] for each in sample[2]]
                
                input_seq_embed1,input_seq_embed2,target_time,target_day,poi_embed,gps_embed= get_rotation_and_loss(sample,label_seq,label_seq_cats,'train')
                
                target_times.append(target_time)
                target_days.append(target_day)
                batch_seq_embed1s.append(input_seq_embed1)
                batch_seq_embed2s.append(input_seq_embed2)
                poi_embeds.append(poi_embed)
                gps_embeds.append(gps_embed)
                
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))

            batch_size = len(batch_input_seqs)
            max_seq_len = max(batch_seq_lens)
            
            batch_padded1 = pad_sequence(batch_seq_embed1s, batch_first=True, padding_value=-1)
            batch_padded2 = pad_sequence(batch_seq_embed2s, batch_first=True, padding_value=-1)
            
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
            batch_target_time = pad_sequence(target_times,batch_first=True,padding_value=-1)
            batch_target_day = pad_sequence(target_days,batch_first=True,padding_value=-1)
            poi_embeds_padded = pad_sequence(poi_embeds,batch_first=True,padding_value=-1)
            gps_embeds_padded = pad_sequence(gps_embeds,batch_first=True,padding_value=-1)

            # Feedforward
            x1 = batch_padded1.to(device=args.device, dtype=torch.float)
            x2 = batch_padded2.to(device=args.device, dtype=torch.float)
            src_mask = seq_model.generate_square_subsequent_mask(x1.size(1)).to(args.device)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)

            y_pred_poi = seq_model(x1,x2,src_mask,batch_target_time,batch_target_day,poi_embeds_padded,gps_embeds_padded)

            # Get loss
            loss_poi = criterion_poi(y_pred_poi.transpose(1, 2), y_poi)

            # Final loss
            #loss = loss_poi + loss_rotate/batch_size
            loss = loss_poi 
            optimizer.zero_grad()

            loss.backward(retain_graph=True)

            optimizer.step()


            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            mrr = 0
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
            train_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            train_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            train_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            train_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            train_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            train_batches_mrr_list.append(mrr / len(batch_label_pois))
            train_batches_loss_list.append(loss.detach().cpu().numpy())
            train_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            print("time11",time.time())
            # Report training progress
            if (b_idx % (25)) == 0:
                sample_idx = 0
                batch_pred_pois_wo_attn = y_pred_poi.detach().cpu().numpy()
                logging.info(f'Epoch:{epoch}, batch:{b_idx}, '
                             f'train_batch_loss:{loss.item():.2f}, '
                             f'train_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                             f'train_move_loss:{np.mean(train_batches_loss_list):.2f}\n'
                             f'train_move_poi_loss:{np.mean(train_batches_poi_loss_list):.2f}\n'
                             f'train_move_top1_acc:{np.mean(train_batches_top1_acc_list):.4f}\n'
                             f'train_move_top5_acc:{np.mean(train_batches_top5_acc_list):.4f}\n'
                             f'train_move_top10_acc:{np.mean(train_batches_top10_acc_list):.4f}\n'
                             f'train_move_top20_acc:{np.mean(train_batches_top20_acc_list):.4f}\n'
                             f'train_move_mAP20:{np.mean(train_batches_mAP20_list):.4f}\n'
                             f'train_move_MRR:{np.mean(train_batches_mrr_list):.4f}\n'
                             f'traj_id:{batch[sample_idx][0][0]}\n'
                             f'input_seq: {batch[sample_idx][1]}\n'
                             f'label_seq:{batch[sample_idx][2]}\n'
                             f'pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             )
            print("time12",time.time())
            # print("time 13",time.time())

        # train end --------------------------------------------------------------------------------------------------------
        user_embed_model.eval()
        poi_embed_model.eval()

        time_embed_model_user.eval()
        time_embed_model_user_tgt.eval()
        
        time_embed_model_poi.eval()
        time_embed_model_poi_tgt.eval()
        
        time_embed_model_poi_day.eval()
        time_embed_model_poi_day_tgt.eval()
        
        time_embed_model_user_day.eval()
        time_embed_model_user_day_tgt.eval()
        
        seq_model.eval()


        gps_embed_model.eval()
        gps_encoder.eval()
        
        
        val_batches_top1_acc_list = []
        val_batches_top5_acc_list = []
        val_batches_top10_acc_list = []
        val_batches_top20_acc_list = []
        val_batches_mAP20_list = []
        val_batches_mrr_list = []
        val_batches_loss_list = []
        val_batches_poi_loss_list = []
        val_batches_time_loss_list = []
        val_batches_cat_loss_list = []
        for vb_idx, batch in enumerate(val_loader):

            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_labels_poi = []


            target_times = []
            target_days = []
            batch_seq_embed1s = []
            batch_seq_embed2s = []
            gps_embeds = []
            poi_embeds = []

            # Convert input seq to embeddings
            for sample in batch:
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                label_seq_cats = [each[2] for each in sample[2]]
                input_seq_embed1,input_seq_embed2,target_time,target_day,poi_embed,gps_embed= get_rotation_and_loss(sample,label_seq,label_seq_cats,mode='test')
                
                target_times.append(target_time)
                target_days.append(target_day)
                batch_seq_embed1s.append(input_seq_embed1)
                batch_seq_embed2s.append(input_seq_embed2)
                poi_embeds.append(poi_embed)
                gps_embeds.append(gps_embed)
                
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                
            batch_size = len(batch_input_seqs)
            max_seq_len = max(batch_seq_lens)

            # Pad seqs for batch training
            batch_padded1 = pad_sequence(batch_seq_embed1s, batch_first=True, padding_value=-1)
            batch_padded2 = pad_sequence(batch_seq_embed2s, batch_first=True, padding_value=-1)
            
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
            batch_target_time = pad_sequence(target_times,batch_first=True,padding_value=-1)
            batch_target_day = pad_sequence(target_days,batch_first=True,padding_value=-1)
            poi_embeds_padded = pad_sequence(poi_embeds,batch_first=True,padding_value=-1)
            gps_embeds_padded = pad_sequence(gps_embeds,batch_first=True,padding_value=-1)

            # Feedforward
            x1 = batch_padded1.to(device=args.device, dtype=torch.float)
            x2 = batch_padded2.to(device=args.device, dtype=torch.float)
            src_mask = seq_model.generate_square_subsequent_mask(x1.size(1)).to(args.device)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_pred_poi = seq_model(x1,x2,src_mask,batch_target_time,batch_target_day,poi_embeds_padded,gps_embeds_padded)

            # Calculate loss
            loss_poi = criterion_poi(y_pred_poi.transpose(1, 2), y_poi)
            loss = loss_poi
            # loss = loss_poi  + loss_cat
            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            mrr = 0
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
            val_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            val_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            val_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            val_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            val_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            val_batches_mrr_list.append(mrr / len(batch_label_pois))
            val_batches_loss_list.append(loss.detach().cpu().numpy())
            val_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())

            # Report validation progress
            if (vb_idx % (20)) == 0:
                sample_idx = 0
                batch_pred_pois_wo_attn = y_pred_poi.detach().cpu().numpy()
                logging.info(f'Epoch:{epoch}, batch:{vb_idx}, '
                             f'val_batch_loss:{loss.item():.2f}, '
                             f'val_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                             f'val_move_loss:{np.mean(val_batches_loss_list):.2f} \n'
                             f'val_move_poi_loss:{np.mean(val_batches_poi_loss_list):.2f} \n'
                             f'val_move_top1_acc:{np.mean(val_batches_top1_acc_list):.4f} \n'
                             f'val_move_top5_acc:{np.mean(val_batches_top5_acc_list):.4f} \n'
                             f'val_move_top10_acc:{np.mean(val_batches_top10_acc_list):.4f} \n'
                             f'val_move_top20_acc:{np.mean(val_batches_top20_acc_list):.4f} \n'
                             f'val_move_mAP20:{np.mean(val_batches_mAP20_list):.4f} \n'
                             f'val_move_MRR:{np.mean(val_batches_mrr_list):.4f} \n'
                             f'traj_id:{batch[sample_idx][0]}\n'
                             f'input_seq:{batch[sample_idx][1]}\n'
                             f'label_seq:{batch[sample_idx][2]}\n'
                             f'pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             )
        # valid end --------------------------------------------------------------------------------------------------------

        # Calculate epoch metrics
        epoch_train_top1_acc = np.mean(train_batches_top1_acc_list)
        epoch_train_top5_acc = np.mean(train_batches_top5_acc_list)
        epoch_train_top10_acc = np.mean(train_batches_top10_acc_list)
        epoch_train_top20_acc = np.mean(train_batches_top20_acc_list)
        epoch_train_mAP20 = np.mean(train_batches_mAP20_list)
        epoch_train_mrr = np.mean(train_batches_mrr_list)
        epoch_train_loss = np.mean(train_batches_loss_list)
        epoch_train_poi_loss = np.mean(train_batches_poi_loss_list)

        epoch_val_top1_acc = np.mean(val_batches_top1_acc_list)
        epoch_val_top5_acc = np.mean(val_batches_top5_acc_list)
        epoch_val_top10_acc = np.mean(val_batches_top10_acc_list)
        epoch_val_top20_acc = np.mean(val_batches_top20_acc_list)
        epoch_val_mAP20 = np.mean(val_batches_mAP20_list)
        epoch_val_mrr = np.mean(val_batches_mrr_list)
        epoch_val_loss = np.mean(val_batches_loss_list)
        epoch_val_poi_loss = np.mean(val_batches_poi_loss_list)

        # Save metrics to list
        train_epochs_loss_list.append(epoch_train_loss)
        train_epochs_poi_loss_list.append(epoch_train_poi_loss)

        train_epochs_top1_acc_list.append(epoch_train_top1_acc)
        train_epochs_top5_acc_list.append(epoch_train_top5_acc)
        train_epochs_top10_acc_list.append(epoch_train_top10_acc)
        train_epochs_top20_acc_list.append(epoch_train_top20_acc)
        train_epochs_mAP20_list.append(epoch_train_mAP20)
        train_epochs_mrr_list.append(epoch_train_mrr)
        val_epochs_loss_list.append(epoch_val_loss)
        val_epochs_poi_loss_list.append(epoch_val_poi_loss)

        val_epochs_top1_acc_list.append(epoch_val_top1_acc)
        val_epochs_top5_acc_list.append(epoch_val_top5_acc)
        val_epochs_top10_acc_list.append(epoch_val_top10_acc)
        val_epochs_top20_acc_list.append(epoch_val_top20_acc)
        val_epochs_mAP20_list.append(epoch_val_mAP20)
        val_epochs_mrr_list.append(epoch_val_mrr)

        # Monitor loss and score
        monitor_loss = epoch_val_loss
        monitor_score = np.mean(epoch_val_top1_acc * 4 + epoch_val_top20_acc)

        # Learning rate schuduler
        #lr_scheduler.step(monitor_loss)
        lr_scheduler.step(epoch)

        # Print epoch results
        logging.info(f"Epoch {epoch}/{args.epochs}\n"
                     f"train_loss:{epoch_train_loss:.4f}, "
                     f"train_poi_loss:{epoch_train_poi_loss:.4f}, "

                     f"train_top1_acc:{epoch_train_top1_acc:.4f}, "
                     f"train_top5_acc:{epoch_train_top5_acc:.4f}, "
                     f"train_top10_acc:{epoch_train_top10_acc:.4f}, "
                     f"train_top20_acc:{epoch_train_top20_acc:.4f}, "
                     f"train_mAP20:{epoch_train_mAP20:.4f}, "
                     f"train_mrr:{epoch_train_mrr:.4f}\n"
                     f"val_loss: {epoch_val_loss:.4f}, "
                     f"val_poi_loss: {epoch_val_poi_loss:.4f}, "

                     f"val_top1_acc:{epoch_val_top1_acc:.4f}, "
                     f"val_top5_acc:{epoch_val_top5_acc:.4f}, "
                     f"val_top10_acc:{epoch_val_top10_acc:.4f}, "
                     f"val_top20_acc:{epoch_val_top20_acc:.4f}, "
                     f"val_mAP20:{epoch_val_mAP20:.4f}, "
                     f"val_mrr:{epoch_val_mrr:.4f}")
        
        cur_score = epoch_val_top1_acc * 5 + epoch_val_top5_acc + epoch_val_top10_acc + epoch_val_mrr
        if cur_score > best_score:
            best_score = cur_score
            best_score_list = [epoch_val_top1_acc,epoch_val_top5_acc,epoch_val_top10_acc,epoch_val_top20_acc,epoch_val_mrr]

        # Save train/val metrics for plotting purpose
        if (epoch + 1) % 5 == 0 :
            logging.info(f"best result is {best_score_list}")
            with open(os.path.join(args.save_dir, 'metrics-train.txt'), "w") as f:
                print(f'train_epochs_loss_list={[float(f"{each:.4f}") for each in train_epochs_loss_list]}', file=f)
                print(f'train_epochs_poi_loss_list={[float(f"{each:.4f}") for each in train_epochs_poi_loss_list]}', file=f)
                print(f'train_epochs_time_loss_list={[float(f"{each:.4f}") for each in train_epochs_time_loss_list]}',
                    file=f)
                print(f'train_epochs_cat_loss_list={[float(f"{each:.4f}") for each in train_epochs_cat_loss_list]}', file=f)
                print(f'train_epochs_top1_acc_list={[float(f"{each:.4f}") for each in train_epochs_top1_acc_list]}', file=f)
                print(f'train_epochs_top5_acc_list={[float(f"{each:.4f}") for each in train_epochs_top5_acc_list]}', file=f)
                print(f'train_epochs_top10_acc_list={[float(f"{each:.4f}") for each in train_epochs_top10_acc_list]}',
                    file=f)
                print(f'train_epochs_top20_acc_list={[float(f"{each:.4f}") for each in train_epochs_top20_acc_list]}',
                    file=f)
                print(f'train_epochs_mAP20_list={[float(f"{each:.4f}") for each in train_epochs_mAP20_list]}', file=f)
                print(f'train_epochs_mrr_list={[float(f"{each:.4f}") for each in train_epochs_mrr_list]}', file=f)
            with open(os.path.join(args.save_dir, 'metrics-val.txt'), "w") as f:
                print(f'val_epochs_loss_list={[float(f"{each:.4f}") for each in val_epochs_loss_list]}', file=f)
                print(f'val_epochs_poi_loss_list={[float(f"{each:.4f}") for each in val_epochs_poi_loss_list]}', file=f)
                print(f'val_epochs_time_loss_list={[float(f"{each:.4f}") for each in val_epochs_time_loss_list]}', file=f)
                print(f'val_epochs_cat_loss_list={[float(f"{each:.4f}") for each in val_epochs_cat_loss_list]}', file=f)
                print(f'val_epochs_top1_acc_list={[float(f"{each:.4f}") for each in val_epochs_top1_acc_list]}', file=f)
                print(f'val_epochs_top5_acc_list={[float(f"{each:.4f}") for each in val_epochs_top5_acc_list]}', file=f)
                print(f'val_epochs_top10_acc_list={[float(f"{each:.4f}") for each in val_epochs_top10_acc_list]}', file=f)
                print(f'val_epochs_top20_acc_list={[float(f"{each:.4f}") for each in val_epochs_top20_acc_list]}', file=f)
                print(f'val_epochs_mAP20_list={[float(f"{each:.4f}") for each in val_epochs_mAP20_list]}', file=f)
                print(f'val_epochs_mrr_list={[float(f"{each:.4f}") for each in val_epochs_mrr_list]}', file=f)


if __name__ == '__main__':
    args = parameter_parser()
    # The name of node features in NYC/graph_X.csv
    args.feature1 = 'checkin_cnt'
    args.feature2 = 'poi_catid'
    args.feature3 = 'latitude'
    args.feature4 = 'longitude'
    train(args)
