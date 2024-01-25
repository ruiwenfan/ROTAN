import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import MultiheadAttention
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils import rotate_batch
from RetNet.src.xpos_relative_position import XPOS
import numpy as np

import copy


class NodeAttnMap(nn.Module):
    def __init__(self, in_features, nhid, use_mask=False):
        super(NodeAttnMap, self).__init__()
        self.use_mask = use_mask
        self.out_features = nhid
        self.W = nn.Parameter(torch.empty(size=(in_features, nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * nhid, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, X, A):
        Wh = torch.mm(X, self.W)

        e = self._prepare_attentional_mechanism_input(Wh)

        if self.use_mask:
            e = torch.where(A > 0, e, torch.zeros_like(e))  # mask

        A = A + 1  # shift from 0-1 to 1-2
        e = e * A

        return e

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)

        return x


class UserEmbeddings(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
        )

    def forward(self, user_idx):
        embed = self.user_embedding(user_idx)
        return embed
class TimeIntervalSim(nn.Module):
    def __init__(self,features) -> None:
        super(TimeIntervalSim,self).__init__()
        
        self.fea = features
        self.fn = nn.Linear(features,1)
    def forward(self,time):
        return self.fn(time)


class CategoryEmbeddings(nn.Module):
    def __init__(self, num_cats, embedding_dim):
        super(CategoryEmbeddings, self).__init__()

        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats,
            embedding_dim=embedding_dim,
        )

    def forward(self, cat_idx):
        embed = self.cat_embedding(cat_idx)
        return embed


class FuseEmbeddings(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, poi_embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), dim=-1))
        x = self.leaky_relu(x)
        return x


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class OriginUserTime2Vec(nn.Module):
    def __init__(self, activation, out_dim,user_num):
        super(OriginUserTime2Vec, self).__init__()
        self.user_bias = nn.parameter.Parameter(torch.zeros(user_num,out_dim))
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)

    def forward(self, x,user_idx):
        fea = x.view(-1,1)
        pre = self.l1(fea)
        user_prefer = self.user_bias[user_idx]
        pre += user_prefer
        return pre
    
class OriginTime2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(OriginTime2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)

    def forward(self, x):
        fea = x.view(-1,1)
        return self.l1(fea)
    
class Time2Vec(nn.Module):
    def __init__(self,out_dim) -> None:
        super(Time2Vec,self).__init__()
        self.w = nn.parameter.Parameter(torch.randn(1,out_dim))
        self.b = nn.parameter.Parameter(torch.randn(1,out_dim))
        self.f = torch.cos
    
    def forward(self,time):
        """_summary_

        Args:
            time (1d tensor): shape is seq_len 

        Returns:
            time embeddings (2d tensor): shape is seq_len * time_dim
        """
        vec_time = time.view(-1,1)
        out = torch.matmul(vec_time,self.w) + self.b
        v1 = out[:,0].view(-1,1)
        v2 = out[:,1:]
        v2 = self.f(v2)
        return torch.cat((v1,v2),dim=-1)
    
class CatTime2Vec(nn.Module):
    def __init__(self,cat_num,out_dim) -> None:
        super(CatTime2Vec,self).__init__()
        self.w0 = nn.parameter.Parameter(torch.randn(cat_num, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(cat_num, 1))
        
        self.w = nn.parameter.Parameter(torch.randn(cat_num, out_dim - 1))
        self.b = nn.parameter.Parameter(torch.randn(cat_num, out_dim - 1))
        
    # cat_idx : s
    # norm_time : s
    def forward(self,cat_idx,norm_time):
        w = self.w[cat_idx]
        b = self.b[cat_idx]
        w0 = self.w0[cat_idx]
        b0 = self.b0[cat_idx]

        norm_time_ = norm_time.view(-1,1)
        v1 = torch.sin(norm_time_ * w + b)
        v2 = norm_time_ * w0 + b0
        return torch.cat((v1,v2),dim=-1)

class RightPositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=600):
        super(RightPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, num_poi, embed_size, nhead, nhid, nlayers, target_time_dim,args,dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        from torch.nn import TransformerDecoder,TransformerDecoderLayer
        self.pos_encoder1 = RightPositionalEncoding(args.user_embed_dim+args.poi_embed_dim, dropout)
        self.pos_encoder2 = RightPositionalEncoding(args.poi_embed_dim,dropout)

        encoder_layers1 = TransformerEncoderLayer(args.user_embed_dim+args.poi_embed_dim, nhead, nhid//2, dropout,batch_first=True)
        self.transformer_encoder1 = TransformerEncoder(encoder_layers1, nlayers)
        
        encoder_layers2 = TransformerEncoderLayer(args.poi_embed_dim, nhead, nhid//2, dropout,batch_first=True)
        self.transformer_encoder2 = TransformerEncoder(encoder_layers2, nlayers)
        user_time_dim = int(0.5*(args.user_embed_dim+args.poi_embed_dim))
        
        self.args = args
        
        self.decoder_poi1 = nn.Linear(args.user_embed_dim+2*args.poi_embed_dim, num_poi)
        self.decoder_poi2 = nn.Linear(args.poi_embed_dim+args.gps_embed_dim, num_poi)
        

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def attention_aggregation(self,src,target_time,traj_time):
        seq_len = src.shape[1]
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = ~mask
        mask = mask.to(self.args.device)
        
        out = self.attn(target_time,traj_time,traj_time,need_weights=True,attn_mask=mask)
        return out[1]

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi1.bias.data.zero_()
        self.decoder_poi1.weight.data.uniform_(-initrange, initrange)
        self.decoder_poi2.bias.data.zero_()
        self.decoder_poi2.weight.data.uniform_(-initrange, initrange)
    # target_poi : (b,s,d)
    # neg_poi : (b,s,k,d)
    # src1 : (b,s,d)
    def forward(self, src1,src2, src_mask,target_hour,target_day,poi_embeds,gps_embeds):
        
        src1 = src1 * math.sqrt(self.args.user_embed_dim+self.args.poi_embed_dim)
        src1 = self.pos_encoder1(src1)
        src1 = self.transformer_encoder1(src1,src_mask)
        
        user_time_dim = int(0.5*(self.args.user_embed_dim+self.args.poi_embed_dim))

        #src1_hour = rotate_batch(src1,target_hour[:,:,:user_time_dim],user_time_dim,self.args.device)
        #src1_day = rotate_batch(src1,target_day[:,:,:user_time_dim],user_time_dim,self.args.device)
        #src1_hour = torch.cat((src1,target_hour[:,:,:user_time_dim]),dim=-1)
        #src1_day = torch.cat((src1,target_day[:,:,:user_time_dim]),dim=-1)
        
        #src1 = 0.7*src1_hour + 0.3*src1_day
        src1 = torch.cat((src1,poi_embeds),dim=-1)
        
        out_poi_prob1 = self.decoder_poi1(src1)
        
        src2 = src2 * math.sqrt(self.args.poi_embed_dim)
        src2 = self.pos_encoder2(src2)
        src2 = self.transformer_encoder2(src2,src_mask)
        
        #src2_hour = rotate_batch(src2,target_hour[:,:,user_time_dim:],64,self.args.device)
        #src2_day = rotate_batch(src2,target_day[:,:,user_time_dim:],64,self.args.device)
        #src2_hour = torch.cat((src2,target_hour[:,:,user_time_dim:]),dim=-1)
        #src2_day = torch.cat((src2,target_day[:,:,user_time_dim:]),dim=-1)
        
        #src2 = 0.7*src2_hour + 0.3*src2_day
        src2 = torch.cat((src2,gps_embeds),dim=-1)
        
        out_poi_prob2 = self.decoder_poi2(src2)
        
        
        out_poi_prob = 0.7*out_poi_prob1 + 0.3*out_poi_prob2

        return out_poi_prob

    
class PoiEmbeddings(nn.Module):
    def __init__(self,num_pois,embedding_dim):
        super(PoiEmbeddings,self).__init__()

        self.poi_embedding = nn.Embedding(
            num_embeddings=num_pois,
            embedding_dim=embedding_dim
        )
    def forward(self,poi_idx):
        embed = self.poi_embedding(poi_idx)
        return embed
    
class RotationTime(nn.Module):
    def __init__(self,dim) -> None:
        super(RotationTime,self).__init__()
        self.ln = nn.Linear(2,dim)
    def forward(self,time):
        c = torch.cos(time)
        s = torch.sin(time)
        t = torch.stack([c,s]).t()
        return self.ln(t)
    
class TimeEmbeddings(nn.Module):
    def __init__(self,num_times,embedding_dim) -> None:
        super(TimeEmbeddings,self).__init__()
        
        self.time_embedding = nn.Embedding(
            num_embeddings=num_times,
            embedding_dim=embedding_dim
        )
    def forward(self,time_idx):
        embed = self.time_embedding(time_idx)
        return embed
    
def clones(module, num_sub_layer):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_sub_layer)])


class GPSEmbeddings(nn.Module):
    def __init__(self,num_gps,embedding_dim) -> None:
        super(GPSEmbeddings,self).__init__()
        self.gps_embedding = nn.Embedding(
            num_embeddings=num_gps,
            embedding_dim=embedding_dim
        )
    def forward(self,gps_idx):
        embed = self.gps_embedding(gps_idx)
        return embed

class GPSEncoder(nn.Module):
    def __init__(self,embed_size,nhead,nhid,nlayers,dropout) -> None:
        super(GPSEncoder,self).__init__()
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout,batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size
        self.norm = nn.LayerNorm(embed_size)
    # s*l*d
    def forward(self,src):
        src = src * math.sqrt(self.embed_size)
        x = self.transformer_encoder(src)
        x = torch.mean(x,-2)
        return self.norm(x)

class SimplePredict(nn.Module):
    def __init__(self,input_embed,out_embed) -> None:
        super(SimplePredict,self).__init__()
        self.trans = nn.Linear(input_embed,out_embed)
        
    def forward(self,src,candidate_poi_embeddings):
        src = self.trans(src)
        #print(f'src shape is {src.shape}')
        #print(f'candidate_poi_embeddings is {candidate_poi_embeddings.shape}')
        out_poi = torch.matmul(src,candidate_poi_embeddings.transpose(0,1))
        
        return out_poi

class TimeEncoder(nn.Module):
    r"""
    This is a trainable encoder to map continuous time value into a low-dimension time vector.
    Ref: https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs/blob/master/module.py

    The input of ts should be like [E, 1] with all time interval as values.
    """

    def __init__(self, embedding_dim):
        super(TimeEncoder, self).__init__()
        self.time_dim = embedding_dim
        self.expand_dim = self.time_dim
        self.use_linear_trans = True

        self.basis_freq = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float())
        self.phase = nn.Parameter(torch.zeros(self.time_dim).float())
        if self.use_linear_trans:
            self.dense = nn.Linear(self.time_dim, self.expand_dim, bias=False)
            nn.init.xavier_normal_(self.dense.weight)

    def forward(self, ts):
        if ts.dim() == 1:
            dim = 1
            edge_len = ts.size().numel()
        else:
            edge_len, dim = ts.size()
        ts = ts.view(edge_len, dim)
        map_ts = ts * self.basis_freq.view(1, -1)
        map_ts += self.phase.view(1, -1)
        harmonic = torch.cos(map_ts)
        if self.use_linear_trans:
            harmonic = harmonic.type(self.dense.weight.dtype)
            harmonic = self.dense(harmonic)
        return harmonic
    
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, input_size)
        self.layer2 = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x