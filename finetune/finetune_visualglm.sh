#! /bin/bash
NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
MODEL_TYPE="visualglm-6b"

MODEL_ARGS="--max_source_length 64 \
    --max_target_length 256 \
    --lora_rank 10 \
    --layer_range 0 14 \
    --pre_seq_len 4" 
OPTIONS_SAT="SAT_HOME=$1" #"SAT_HOME=/raid/dm/sat_models"
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

# 选择要用来微调的数据集 (支持 coco 或 flickr30k)
DATASET="flickr30k" 

if [ "$DATASET" = "coco" ]; then
    train_data="./coco_finetune/coco_train.json"
    eval_data="./coco_finetune/coco_val.json"
elif [ "$DATASET" = "flickr30k" ]; then
    train_data="./flickr30k_finetune/flickr30k_train.json"
    eval_data="./flickr30k_finetune/flickr30k_val.json"
else
    echo "未识别的数据集: $DATASET"
    exit 1
fi

gpt_options=" \
       --experiment-name finetune-$MODEL_TYPE-$DATASET \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 6000 \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${train_data} \
       --valid-data ${eval_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup 0.04 \
       --checkpoint-activations \
       --save-interval 3000 \
       --eval-interval 500 \
       --save "./checkpoints" \
       --split 1 \
       --eval-iters 10 \
       --eval-batch-size 8 \
       --lr 0.00005 \
       --batch-size 8 \
       --gradient-accumulation-steps 2 \
       --skip-init \
       --bf16 \
       --use_lora \
       --log-interval 10
"

SAT_HOME_PATH=${1:-"/gemini/code/VGLM/root/.sat_models"}

# 使用 DeepSpeed 运行并注入环境变量
run_cmd="SAT_HOME=${SAT_HOME_PATH} deepspeed --master_port 16666 --num_gpus=1 finetune_visualglm.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
