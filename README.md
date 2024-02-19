# ROTAN
The implementation code of ROTAN.

KG's implementation refers to RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.

## Dataset 

Download three datasets from [Datasets](https://drive.google.com/drive/folders/1xsML0LIhTaF5x0rXmqwLsmwCKabFb-D5?usp=sharing) and move them to dataset/ .

## Pre-trained Graph

Download pre-trained graphs from [models](https://drive.google.com/drive/folders/1qVKTWVWL9qr8-yY7EKv3ZPk2YcEArEC5?usp=sharing) and move them to KG/models .

## Train

`nohup ./train_nyc.sh > nyc.txt &`

`nohup ./train_tky.sh > tky.txt &`

`nohup ./train_ca.sh > ca.txt &`

