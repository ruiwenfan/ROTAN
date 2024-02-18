"""Parsing the parameters."""
import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run ROTAN.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Random seed')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:1',
                        help='')
    parser.add_argument('--ngrams',
                        type=int,
                        default=6,
                        help='N-grams of quadkey encoder'
                        )
    parser.add_argument('--quadkey-len',
                        type=int,
                        default=25,
                        help='length of quadkey encoder'
                        )
    parser.add_argument('--neg-num',
                        type=int,
                        default=10,
                        help='nums of neg sample'
                        )
    # Data
    parser.add_argument('--data-adj-mtx',
                        type=str,
                        default='dataset/NYC/graph_A.csv',
                        help='Graph adjacent path')
    parser.add_argument('--data-node-feats',
                        type=str,
                        default='dataset/NYC/graph_X.csv',
                        help='Graph node features path')
    parser.add_argument('--data-train',
                        type=str,
                        default='dataset/NYC/NYC_train.csv',
                        help='Training data path')
    parser.add_argument('--data-val',
                        type=str,
                        default='dataset/NYC/NYC_val.csv',
                        help='Validation data path')
    parser.add_argument('--short-traj-thres',
                        type=int,
                        default=2,
                        help='Remove over-short trajectory')
    parser.add_argument('--time-units',
                        type=int,
                        default=48,
                        help='Time unit is 0.5 hour, 24/0.5=48')
    parser.add_argument('--time-feature',
                        type=str,
                        default='norm_in_day_time',
                        help='The name of time feature in the data')
    parser.add_argument('--traj-feature',
                        type=str,
                        default='pseudo_session_trajectory_id',
                        help='The traj_id of trajectory')
    parser.add_argument('--poi-pre-embedding',
                        type=str,
                        default='KG/models/qua_RotatE_NYC_time2vec_norm/entity_embedding.npy',
                        help='The traj_id of trajectory')
    parser.add_argument('--time-pre-embedding',
                        type=str,
                        default='KG/models/qua_RotatE_NYC_time2vec_norm/relation_embedding.npy',
                        help='The traj_id of trajectory')
    parser.add_argument('--poi-time-embed-state-dict',
                        type=str,
                        default='KG/models/qua_RotatE_NYC_time2vec_norm/time_embed_model.pth',
                        help='The traj_id of trajectory')
    parser.add_argument('--poi-time-tgt-embed-state-dict',
                        type=str,
                        default='KG/models/qua_RotatE_NYC_time2vec_norm/time_embed_model_tgt.pth',
                        help='The traj_id of trajectory')
    parser.add_argument('--poi-entity',
                        type=str,
                        default='KG/data/NYC/poi_qua/entities.dict',
                        help='The traj_id of trajectory')
    parser.add_argument('--user-pre-embedding',
                        type=str,
                        default='KG/models/RotatE_NYC_mini_user/entity_embedding.npy',
                        help='The traj_id of trajectory')
    parser.add_argument('--user-time-pre-embedding',
                        type=str,
                        default='KG/models/RotatE_NYC_mini_user/relation_embedding.npy',
                        help='The traj_id of trajectory')
    parser.add_argument('--user-entity',
                        type=str,
                        default='KG/data/NYC_mini/user/entities.dict',
                        help='The traj_id of trajectory')


    # Model hyper-parameters
    parser.add_argument('--poi-embed-dim',
                        type=int,
                        default=128,
                        help='POI embedding dimensions')
    parser.add_argument('--gps-embed-dim',
                        type=int,
                        default=128,
                        help='GPS embedding dimensions')
    parser.add_argument('--user-embed-dim',
                        type=int,
                        default=128,
                        help='User embedding dimensions')
    parser.add_argument('--gcn-dropout',
                        type=float,
                        default=0.3,
                        help='Dropout rate for gcn')
    parser.add_argument('--gcn-nhid',
                        type=list,
                        default=[32, 64],
                        help='List of hidden dims for gcn layers')
    parser.add_argument('--transformer-nhid',
                        type=int,
                        default=1024,
                        help='Hid dim in TransformerEncoder')
    parser.add_argument('--transformer-nlayers',
                        type=int,
                        default=2,
                        help='Num of TransformerEncoderLayer')
    parser.add_argument('--transformer-nhead',
                        type=int,
                        default=2,
                        help='Num of heads in multiheadattention')
    parser.add_argument('--transformer-dropout',
                        type=float,
                        default=0.3,
                        help='Dropout rate for transformer')
    parser.add_argument('--time-embed-dim',
                        type=int,
                        default=32,
                        help='Time embedding dimensions')
    parser.add_argument('--cat-embed-dim',
                        type=int,
                        default=32,
                        help='Category embedding dimensions')
    parser.add_argument('--time-loss-weight',
                        type=int,
                        default=10,
                        help='Scale factor for the time loss term')
    parser.add_argument('--node-attn-nhid',
                        type=int,
                        default=128,
                        help='Node attn map hidden dimensions')

    # Training hyper-parameters
    parser.add_argument('--batch',
                        type=int,
                        default=20,
                        help='Batch size.')
    parser.add_argument('--epochs',
                        type=int,
                        default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr',
                        type=float,
                        default=2e-2,
                        help='Initial learning rate.')
    parser.add_argument('--lr-scheduler-factor',
                        type=float,
                        default=0.1,
                        help='Learning rate scheduler factor')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--warmup-lr-init',
                        type=float,
                        default=1e-3,
                        help='Weight decay (L2 loss on parameters).')

    # Experiment config
    parser.add_argument('--save-weights',
                        action='store_true',
                        default=True,
                        help='whether save the model')
    parser.add_argument('--save-embeds',
                        action='store_true',
                        default=False,
                        help='whether save the embeddings')
    parser.add_argument('--workers',
                        type=int,
                        default=0,
                        help='Num of workers for dataloader.')
    parser.add_argument('--project',
                        default='runs/train',
                        help='save to project/name')
    parser.add_argument('--name',
                        default='exp',
                        help='save to project/name')
    parser.add_argument('--exist-ok',
                        action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False, help='Disables CUDA training.')
    parser.add_argument('--mode',
                        type=str,
                        default='client',
                        help='python console use only')
    parser.add_argument('--port',
                        type=int,
                        default=64973,
                        help='python console use only')

    return parser.parse_args()
