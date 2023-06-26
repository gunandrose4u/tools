num_gpus=8
# torchrun --nproc-per-node=$num_gpus --master-port 29501 runner.py -m facebook/opt-1.3b -t 10 -b 4 -s 512 --max_new_tokens 12
# deepspeed --num_gpus $num_gpus --master-port 29501 runner.py -m facebook/opt-1.3b -t 10 -b 4 -s 512 --max_new_tokens 12

# python runner.py -m facebook/opt-1.3b -t 3 -b 1 --max_new_tokens 1 --token_len 128 --dtype float32

# python runner.py -m decapoda-research/llama-7b-hf -t 3 -b 30 --max_new_tokens 12 --token_len 128
# python runner.py -m decapoda-research/llama-13b-hf -t 3 -b 30 --max_new_tokens 12 --token_len 128

# python runner.py -m t5-3b -t 3 -b 30 --max_new_tokens 24 --token_len 128 --padding_side left
# python runner.py -m EleutherAI/gpt-j-6B -t 3 -b 30 --max_new_tokens 32 --token_len 128 --pad_token_id 50256

#small models
for model in facebook/opt-1.3b t5-3b EleutherAI/gpt-j-6B decapoda-research/llama-7b-hf decapoda-research/llama-13b-hf
do
for b in 1 16
do
for max_new_tokens in 2 128
do
for seq_len in 128 512 1024
do
for num_beams in 1 4
do
echo "model: $model, batch_size: $b, token_len: $seq_len, max_new_tokens: $max_new_tokens, num_beams: $num_beams"
python runner.py -m $model -t 30 -b $b --max_new_tokens $max_new_tokens --seq_len $seq_len  --num_beams $num_beams --use_cache --token_metrics
done
done
done
done
done



# large model
# for model in decapoda-research/llama-30b-hf decapoda-research/llama-65b-hf microsoft/bloom-deepspeed-inference-fp16
#     do
#     for b in 1 16
#         do
#         for max_new_tokens in 1  1024
#             do
#             for seq_len in 512  8192
#                 do
#                     for num_beams in 1
#                         echo "model: $model, batch_size: $b, token_len: $seq_len, max_new_tokens: $max_new_tokens, num_beams: $num_beams"
#                         deepspeed --num_gpus $num_gpus runner.py -m $model -t 30 -b $b --max_new_tokens $max_new_tokens --seq_len $seq_len  --num_beams $num_beams --use_cache --token_metrics --use_kernel
#                     done
#                 done
#             done
#         done
#     done
