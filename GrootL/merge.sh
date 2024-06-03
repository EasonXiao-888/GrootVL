path=path-to-exp
out_path=path-to-out
base_model_130m=mamba-130m-hf
cd ${path} && python zero_to_fp32.py . pytorch_model.bin
cd GrootVL/GrootL
python merge_lora_weights_and_save_hf_model.py \
        --base_model ${base_model_790m} \
        --peft_model ${path} \
        --context_size 2048 \
        --save_path ${out_path}