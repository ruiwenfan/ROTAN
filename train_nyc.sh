export CUDA_DEVICE_ORDER="PCI_BUS_ID"
python old_train.py --data-train dataset/NYC/NYC_train.csv \
                --data-val dataset/NYC/NYC_val.csv \
                --time-units 96 --time-feature local_time \
                --poi-embed-dim 128 --user-embed-dim 128 \
                --time-embed-dim 64 --cat-embed-dim 128 \
                --node-attn-nhid 128     \
                --transformer-nhid 1024 \
                --device cuda:3 \
                --neg-num 80 --lr 1e-3 --warmup-lr-init 1e-4 \
                --transformer-nlayers 4 --transformer-nhead 2 \
                --batch 128 --epochs 60 --name exp1 --weight_decay 8e-4