import math
import numpy as np

from deepspeed.accelerator import get_accelerator
from data_loader import DataLoader
from megatron import get_tokenizer
from pt_hf_nlp import INPUTS


class MegatronNlpDataloader(DataLoader):
    def __init__(self, run_config):
        super().__init__(run_config)

        self._tokenizer = get_tokenizer()
        meet_seq_len_inputs = []

        input_sentences = INPUTS
        if run_config.batch_size > len(INPUTS):
            input_sentences *= math.ceil(run_config.batch_size / len(input_sentences))

        for sen in input_sentences:
            token = self._tokenizer.tokenize(sen)
            # repeat the token to seq_len if needed; otherwise truncate to seq_len
            rp_times = math.ceil(run_config.seq_len / len(token))
            new_token = token * rp_times
            new_token = new_token[: run_config.seq_len]
            meet_seq_len_inputs.append(new_token)

        if len(meet_seq_len_inputs) >= run_config.total_sample_count:
            self._loaded_data_x = meet_seq_len_inputs[: run_config.total_sample_count]
        else:
            for _ in range(run_config.total_sample_count):
                self._loaded_data_x.append(meet_seq_len_inputs[np.random.randint(len(meet_seq_len_inputs))])

    def make_batch(self, data_array, selected_indices):
        return [data_array[i] for i in selected_indices]


class BenchmarkDataLoader(MegatronNlpDataloader):
    def __init__(self, run_config):
        super(BenchmarkDataLoader, self).__init__(run_config)
