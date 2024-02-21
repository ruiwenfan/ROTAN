# ROTAN
The implementation code of ROTAN.

KG's implementation refers to RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.

## Environment

`conda env create -f environment.yml`

## Dataset 

Download three datasets from [Datasets](https://drive.google.com/drive/folders/1xsML0LIhTaF5x0rXmqwLsmwCKabFb-D5?usp=sharing) and create NYC, TKY and CA folder in dataset/ . Then, move three datasets to specified folder.(For example, move NYC_train.csv and NYC_val.csv to dataset/NYC).

## Pre-trained Graph

Download pre-trained graphs from [Models](https://drive.google.com/drive/folders/1qVKTWVWL9qr8-yY7EKv3ZPk2YcEArEC5?usp=sharing) and move them to KG/models .

## Train

`nohup ./train_nyc.sh > nyc.txt &`

`nohup ./train_tky.sh > tky.txt &`

`nohup ./train_ca.sh > ca.txt &`

It should be noted that the CA dataset may be OOM(Out Of Memory). So in line 188 and 272 of [old_train.py](https://github.com/ruiwenfan/ROTAN/blob/main/old_train.py), we restrict the length of trajectory.

`if len(input_seq) < args.short_traj_thres or len(input_seq) >100:`

`if len(input_seq) < args.short_traj_thres or len(input_seq) > 100:`

In NYC and TKY dataset, you can remove `len(input_seq) >100`.




