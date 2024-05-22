## Introduction
   A inference benchmark tool, you can base on this tool to extend usage for different benchmark purpose. You can add a dataloader to provide your dataset for benchmarking, and also you can add your backend for benchmarking on different framework or models.

This tool provides several basic benchmarkers, like mlperf, direct, nlp_generative.

## Usage
### Install python packages
        pip install -r requirement.txt
### Benchmark on single gpu
        python runner.py -m facebook/opt-1.3b -b 1 -s 16
### Benchmark on multiple gpus
        deepspeed --num_gpus 2 runner.py -m facebook/opt-1.3b -b 1 -s 16

## Builtin supported models
    - facebook/opt-1.3b
    - t5-3b
    - EleutherAI/gpt-j-6B
    - decapoda-research/llama-7b-hf
    - decapoda-research/llama-13b-hf
    - decapoda-research/llama-30b-hf
    - decapoda-research/llama-65b-hf 
    - bigscience/bloom-7b1
    - bigscience/bloom
    - microsoft/bloom-deepspeed-inference-fp16

## For more details info
    python runner.py -h

## TTS Reader
### Install 
        conda create -n tts_reader python==3.10 -y
        conda activate tts_reader
        install.cmd
### Run
        python tts_reader_gui.py
### Notice
        For fisrt time run, tts_reader need download model from internet, it will take times base on your network condition. 