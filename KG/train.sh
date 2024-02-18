CUDA_VISIBLE_DEVICES=1 python -u codes/run.py --do_train \
 --cuda \
 --data_path data/CA/poi_qua \
 --model qua_RotatE \
 -n 256 -b 1024 -d 64 \
 -g 24.0 -a 1.0 -adv \
 -lr 0.0001 --max_steps 150000 \
 -save models/qua_RotatE_CA_clone_time2vec_norm --test_batch_size 16 -de