import os
import sys
import random
import numpy as np
import traceback
import datetime


if "OMP_NUM_THREADS" not in os.environ:
    #os.environ["OMP_NUM_THREADS"] = str(10)
    pass

import torch
#torch.set_num_threads(10)


from anubis_logger import logger
from utilities import parse_arguments, save_dict_to_csv
from time import perf_counter
from transformers import BertConfig, BertForQuestionAnswering

def benchmark(model_path, data_path, test_times, csv_path):
    config_json = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "type_vocab_size": 2,
        "vocab_size": 30522
        }

    config = BertConfig(
        attention_probs_dropout_prob=config_json["attention_probs_dropout_prob"],
        hidden_act=config_json["hidden_act"],
        hidden_dropout_prob=config_json["hidden_dropout_prob"],
        hidden_size=config_json["hidden_size"],
        initializer_range=config_json["initializer_range"],
        intermediate_size=config_json["intermediate_size"],
        max_position_embeddings=config_json["max_position_embeddings"],
        num_attention_heads=config_json["num_attention_heads"],
        num_hidden_layers=config_json["num_hidden_layers"],
        type_vocab_size=config_json["type_vocab_size"],
        vocab_size=config_json["vocab_size"])
    model = BertForQuestionAnswering(config)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    logger.debug(f"Model {model}")
    logger.debug(f"Number of parameters {model.num_parameters()}")

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    max_batch_size = 4
    total_samples = 198
    batch_size = 8

    feeds = np.load(data_path, allow_pickle=True)
    x = feeds[0]

    data_x = [x for i in range(1024)]

    run_durations = []
    with torch.no_grad():
        idx = random.randint(0, 1000)
        feed = data_x[idx]
        #print(np.array([np.squeeze(feed['input_ids'], axis=0) for i in range(batch_size)]))
        for i in range(1):
            model.forward(
                input_ids=torch.from_numpy(np.array([np.squeeze(feed['input_ids'], axis=0) for i in range(batch_size)])).to(device),
                attention_mask=torch.from_numpy(np.array([np.squeeze(feed['input_mask'], axis=0) for i in range(batch_size)])).to(device),
                token_type_ids=torch.from_numpy(np.array([np.squeeze(feed['segment_ids'], axis=0) for i in range(batch_size)])).to(device))

        i_total_samples = total_samples
        t2_start = perf_counter()
        for i in range(test_times):
            if total_samples > max_batch_size:
                batch_size = max_batch_size
            else:
                batch_size = total_samples

            idx = random.randint(0, 1000)
            feed = data_x[idx]
            t1_start = perf_counter()
            output = model.forward(
                input_ids=torch.from_numpy(np.array([np.squeeze(feed['input_ids'], axis=0) for i in range(batch_size)])).to(device),
                attention_mask=torch.from_numpy(np.array([np.squeeze(feed['input_mask'], axis=0) for i in range(batch_size)])).to(device),
                token_type_ids=torch.from_numpy(np.array([np.squeeze(feed['segment_ids'], axis=0) for i in range(batch_size)])).to(device))
            t1_end = perf_counter()
            run_durations.append(t1_end-t1_start)

            total_samples = total_samples - batch_size
            if total_samples == 0:
                break

            #if isinstance(output, torch.Tensor):
            #    output = output.cpu().numpy()
        t2_end = perf_counter()
        qps = i_total_samples / (t2_end - t2_start)


        res_benchmark = {}
        res_benchmark['time'] = str(datetime.datetime.now())
        res_benchmark['model'] = "bert-squad"
        res_benchmark['min'] = str(np.min(run_durations))
        res_benchmark['max'] = str(np.max(run_durations))
        res_benchmark['mean'] = str(np.mean(run_durations))
        res_benchmark['50pt'] = str(np.percentile(run_durations, 50))
        res_benchmark['90pt'] = str(np.percentile(run_durations, 90))
        res_benchmark['95pt'] = str(np.percentile(run_durations, 95))
        res_benchmark['99pt'] = str(np.percentile(run_durations, 99))
        res_benchmark['99.9pt'] = str(np.percentile(run_durations, 99.9))
        res_benchmark['qps'] = str(qps)
        res_benchmark['var'] = str(np.std(run_durations) / np.mean(run_durations))
        res_benchmark['std'] = str(np.std(run_durations))
        res_benchmark['framework'] = f"torch+{torch.__version__}"
        res_benchmark['backend'] = "N/A"
        res_benchmark['test_times'] = str(test_times)
        res_benchmark['num_threads'] = "-1"
        res_benchmark['max_batch_size'] = str(max_batch_size)


        logger.info(res_benchmark)
        logger.info("Benchmark done")

        save_dict_to_csv(res_benchmark, csv_path)

def main():
    try:
        args = parse_arguments()
        logger.info(args)
    except:
        return

    benchmark(args.model_path, args.data, args.test_times, args.result_csv)

if __name__ == '__main__':
    try:
        main()
    except:
        logger.error(traceback.format_exc())
        sys.exit(-1)
