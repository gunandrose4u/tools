num_gpus=4
# torchrun --nproc-per-node=$num_gpus --master-port 29501 runner.py -m facebook/opt-1.3b -t 10 -b 4 -s 512 --max_new_tokens 12
# deepspeed --num_nodes=$num_gpus --master-port 29501 runner.py -m facebook/opt-1.3b -t 10 -b 4 -s 512 --max_new_tokens 12
# deepspeed --num_gpus=$num_gpus 
# python runner.py -m facebook/opt-1.3b -t 3 -b 1 --max_new_tokens 1 --token_len 128 --dtype float32

# python runner.py -m decapoda-research/llama-7b-hf -t 3 -b 30 --max_new_tokens 12 --token_len 128
# python runner.py -m decapoda-research/llama-13b-hf -t 3 -b 30 --max_new_tokens 12 --token_len 128

# python runner.py -m t5-3b -t 3 -b 30 --max_new_tokens 24 --token_len 128 --padding_side left 
# python runner.py -m EleutherAI/gpt-j-6B -t 3 -b 30 --max_new_tokens 32 --token_len 128 --pad_token_id 50256

for model in facebook/opt-1.3b 
do
for b in 1 32
do
for max_new_tokens in 1 32 
do
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
    echo "model: $model, batch_size: $b, token_len: $token_len, max_new_tokens: $max_new_tokens, greedy: $greedy, use_cache: $use_cache"
    python runner.py -m $model -t 30 -b $b --max_new_tokens $max_new_tokens --seq_len 128 --greedy --use_cache --token_record
done
done
done
done


for model in facebook/opt-1.3b 
do
for b in 1 32
do
for max_new_tokens in 1 32 
do
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
    echo "model: $model, batch_size: $b, token_len: $token_len, max_new_tokens: $max_new_tokens, greedy: $greedy, use_cache: $use_cache"
    python runner.py -m $model -t 30 -b $b --max_new_tokens $max_new_tokens --seq_len 128 --greedy --use_cache
done
done
done
done

# python runner.py -m facebook/opt-1.3b -t 30 -b 1 --max_new_tokens 16 --token_len 128
# python runner.py -m facebook/opt-1.3b -t 30 -b 8 --max_new_tokens 16 --token_len 256
# python runner.py -m facebook/opt-1.3b -t 30 -b 16 --max_new_tokens 16 --token_len 128
# python runner.py -m facebook/opt-1.3b -t 30 -b 32 --max_new_tokens 16 --token_len 128
# && python runner.py -m decapoda-research/llama-7b-hf -t 30 -b 30 --max_new_tokens 12 --token_len 128



