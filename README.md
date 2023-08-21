# Baseline Project for Defect Views dataset

## Dataset
Your own dataset can be built by inheriting from the abstract class `CustomDataset` in `src.dataset.dataset`

## Config
### General
```
"experiment_name": str. Required for wandb; set to "disabled" to disable it.
```

### Dataset
```
"dataset_path": str. Path to the dataset,
"dataset_type": {`opt6`, `opt_bckg`, `binary`, `qplusv1`, `qplusv2`},
"dataset_splits": List[float] (1 for train/test (e.g. [0.8]), 3 for train/val/test),
"crop_size": int,
"image_size": int. Reshape to image_size,
"augment_online": List[str] (classes that undergo online augmentation),
"augment_offline": List[str] (classes that undergo offline augmentation),
"dataset_mean": List[float] (Grayscale/RGB),
"dataset_std": List[float] (Grayscale/RGB)
```

### Model
```
"model": { "default", "mlp", "cnn", "cnn105", "resnet50", "hrnet_w18", "vit_tiny_patch16_224" } and compare version,
"fsl": Optional if ProtoNet is required (the model must not contain 'compare' in its name)
  "episodes": int,
  "train_n_way": int,
  "train_k_shot_s": int
  "train_k_shot_q": int,
  "test_n_way": int,
  "test_k_shot_s": int,
  "test_k_shot_q": int,
  "enhancement": Optional[str]. { "ipn" }
```

### Train/Test
```
"epochs": int,
"batch_size": int,
"model_test_path": Optional[str]. Define only if you have a saved model and do not want to train again.
"learning_rate": Optional[str],
"optimizer": Optional[str]
```

## Docker
If the use of Docker is required or desired, run the following two commands after installing Docker on the host machine:  
`$ docker compose build`  
`$ docker run --name <image_name> -v <host_path>:<docker_path> lollo/protonet`

The first command reads both the `docker-compose.yml` file and the `Dockerfile` in the main directory, so check them out and perform the necessary changes before running it.

The second command allows the mounting of the volume from the host machine to the docker container. In my case:   
- host path is `/media/lorenzo/M/datasets/dataset_opt/2.3_dataset_opt`
- docker path is `/media/` (in the docker [Linux] image)

This holds true if the dataset path specified in `config.json` is `/media/views/img`. In essence, you are mounting the **content** of `2.3_dataset_opt` folder into the docker's `/media` directory. Hence, it will finally contain the `views` folder with its subdirectories.

## Models
[ProtoNet implementation used](https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch).

# References
```bib
@article{DBLP:journals/corr/SnellSZ17,
  author    = {Jake Snell and
               Kevin Swersky and
               Richard S. Zemel},
  title     = {Prototypical Networks for Few-shot Learning},
  journal   = {CoRR},
  volume    = {abs/1703.05175},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.05175},
  archivePrefix = {arXiv},
  eprint    = {1703.05175},
  timestamp = {Wed, 07 Jun 2017 14:41:38 +0200},
  biburl    = {http://dblp.org/rec/bib/journals/corr/SnellSZ17},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```
