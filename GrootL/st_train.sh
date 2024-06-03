export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=INFO
model_path=mamba-130m-hf

torchrun --nproc_per_node=1 \
        --master_addr=127.0.0.1 \
        --master_port=29500 supervised-fine-tune.py  \
        --model_name_or_path ${model_path} \
        --bf16 True \
        --output_dir out_exp   \
        --model_max_length 2048 \
        --use_flash_attn False \
        --data_path benchmarks/alpaca_data.json \
        --low_rank_training True \
        --num_train_epochs 1  \
        --per_device_train_batch_size 1     \
        --per_device_eval_batch_size 2     \
        --gradient_accumulation_steps 16     \
        --evaluation_strategy "no"     \
        --save_strategy "steps"     \
        --save_steps 100     \
        --save_total_limit 2     \
        --learning_rate 2e-5     \
        --weight_decay 0.0     \
        --warmup_steps 20     \
        --lr_scheduler_type "constant_with_warmup"     \
        --logging_steps 1     \
        --deepspeed "ds_configs/stage2.json" 
