# Run this script to download the data and model checkpoints for the benchmark.
cd $(dirname $0)

# Source: https://github.com/microsoft/Megatron-DeepSpeed/blob/main/dataset/download_vocab.sh
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

# Source: https://github.com/microsoft/Megatron-DeepSpeed/blob/main/dataset/download_ckpt.sh
mkdir -p checkpoints/gpt2_345m
cd checkpoints/gpt2_345m
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip \
    -O megatron_lm_345m_v0.0.zip
unzip megatron_lm_345m_v0.0.zip
rm megatron_lm_345m_v0.0.zip
