# ROTAN
The implementation code of ROTAN.

KG's implementation refers to RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.

## Environment

`conda env create -f environment.yml`

## Dataset 

Create dataset folder and download three datasets from [Datasets](https://github.com/ruiwenfan/ROTAN)

## Pre-trained Graph

Download pre-trained graphs from [models](https://github.com/ruiwenfan/ROTAN) and move them to KG/models

## Train

`nohup ./train_nyc.sh > nyc.txt &`

`nohup ./train_tky.sh > tky.txt &`

`nohup ./train_ca.sh > ca.txt &`

