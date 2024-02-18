export CUDA_DEVICE_ORDER="PCI_BUS_ID"
python old_train.py --data-train dataset/TKY/TKY_train.csv \
                --data-val dataset/TKY/TKY_val.csv \
                --time-units 96 --time-feature local_time \
                --poi-embed-dim 128 --user-embed-dim 128 \
                --time-embed-dim 64 --cat-embed-dim 128 \
                --poi-pre-embedding KG/models/qua_RotatE_TKY_time2vec_norm/entity_embedding.npy \
                --time-pre-embedding KG/models/qua_RotatE_TKY_time2vec_norm/relation_embedding.npy \
                --poi-entity KG/data/TKY/poi_qua/entities.dict \
                --user-pre-embedding KG/models/RotatE_TKY_mini_1_user/entity_embedding.npy \
                --user-time-pre-embedding KG/models/RotatE_TKY_mini_1_user/relation_embedding.npy \
                --user-entity KG/data/TKY_mini1/user/entities.dict \
                --node-attn-nhid 128 \
                --traj-feature trajectory_id \
                --device cuda:0 \
                --neg-num 100 --lr 1e-3 --warmup-lr-init 1e-4 \
                --transformer-nhid 1024 \
                --transformer-nlayers 4 --transformer-nhead 2 \
                --batch 128 --epochs 60 --name exp1 --weight_decay 8e-4