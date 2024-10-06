# Video-DDBM
3D DDBM model

## Building docker image

```
cd docker
./BUILD_DOCKER_IMAGE.sh
./RUN_DOCKER_CONTAINER.sh
docker exec -it ${USER}_video_ddbm zsh
cd workspace
```


## Train

Normal training:

```
export WANDB_API_KEY=****
git config --global --add safe.directory /root/workspace
python3 train.py -e 10000 -b 16 --input_folder_path data/ct --target_folder_path data/mri --image_size 128 --image_depth 128 --save-per-epoch 500
```

DeepSpeed training:

```
export WANDB_API_KEY=****
git config --global --add safe.directory /root/workspace
pip install ninja 

python3 train_deepspeed.py -e 10000 --input_folder_path data/ct --target_folder_path data/mri --image_size 128 --image_depth 128 --save-per-epoch 500
```

set batchsize in `./configs/ds_config.json`
