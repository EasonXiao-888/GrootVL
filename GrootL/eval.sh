export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

path=/Users/easonxiao/Desktop/GrootVL/GrootL/grootl-130m-hf

python lm_harness_eval.py --model mamba \
    --model_args pretrained=${path} \
    --tasks arc_easy,winogrande,openbookqa,sst \
    --device cuda --batch_size 1