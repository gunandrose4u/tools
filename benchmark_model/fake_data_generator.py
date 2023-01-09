import os
import sys
import argparse
import traceback
import yaml

import numpy as np

from anubis_logger import logger


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="Config for fake dataset",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=False,
        type=str,
        default='fake_data.npy',
        help="Generated fake data as npy",
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    fake_data_config = yaml.safe_load(open(args.config, "r").read())
    input_feed = {}
    for input_name in fake_data_config:
        logger.info(f"{input_name}={fake_data_config[input_name]}")
        if 'method' in fake_data_config[input_name]:
            init_method = fake_data_config[input_name]['method']
        else:
            init_method = 'random'

        ishape = fake_data_config[input_name]['sharp']
        dtype = fake_data_config[input_name]['dtype']

        if init_method == 'random':
            input_feed[input_name] = np.random.uniform(size=ishape).astype(dtype).reshape(ishape)
        elif init_method == 'one':
             input_feed[input_name] = np.ones(shape=ishape).astype(dtype).reshape(ishape)
        elif init_method == 'zero':
             input_feed[input_name] = np.zeros(shape=ishape).astype(dtype).reshape(ishape)
        elif init_method == 'arange':
             input_feed[input_name] = np.arange(0, np.prod(ishape), dtype=dtype).reshape(ishape)
        else:
            assert 0, f'not support init_method {init_method}'

    np.save(args.output, [input_feed], allow_pickle=True)

if __name__ == '__main__':
    main()