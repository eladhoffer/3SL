NAME=resnet50_mlperf
LOGDIR="/3SL/results"
DATASETS="$HOME/Datasets"

docker run -it --rm \
    --name $NAME \
    -v "$(pwd)"/results:$LOGDIR \
    -v $DATASETS:/root/Datasets \
    --gpus all \
    -p 0.0.0.0:6006:6006 \
    eladhoffer/3sl /bin/bash -c "\
        tensorboard --logdir=$LOGDIR & \
        python run.py name=$NAME \
                      experiment=imagenet/mlperf_resnet50 \
                      data_dir=/root/Datasets \
    "