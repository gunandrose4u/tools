import math
import torch
import numpy as np

from data_loader import DataLoader
from transformers import AutoTokenizer, LlamaTokenizer
from anubis_logger import logger

LONG_RAW_TEXT = """A quantum computer is a computer that exploits quantum mechanical phenomena.
At small scales, physical matter exhibits properties of both particles and waves,
and quantum computing leverages this behavior using specialized hardware. Classical physics
cannot explain the operation of these quantum devices, and a scalable quantum computer could
perform some calculations exponentially faster than any modern "classical" computer. In particular,
a large-scale quantum computer could break widely used encryption schemes and aid physicists in
performing physical simulations; however, the current state of the art is largely experimental and impractical.
The basic unit of information in quantum computing is the qubit, similar to the bit in traditional
digital electronics. Unlike a classical bit, a qubit can exist in a superposition of its two "basis"
states, which loosely means that it is in both states simultaneously. When measuring a qubit,
the result is a probabilistic output of a classical bit. If a quantum computer manipulates the qubit
in a particular way, wave interference effects can amplify the desired measurement results. The design of
quantum algorithms involves creating procedures that allow a quantum computer to perform calculations efficiently
and quickly."""

INPUTS = [
        LONG_RAW_TEXT,
         "DeepSpeed is a machine learning framework",
         "He is working on",
         "He has a",
         "He got all",
         "Everyone is happy and I can",
         "The new movie that got Oscar this year",
         "In the far far distance from our galaxy,",
         "Peace is the only way"
]

# LLAMA tokenizer has a bug, can not load by AutoTokenizer. And LlamaTokenizer batch token will lead inference failure.
LLAMA_MODEL_NAMES = [
    'decapoda-research/llama-7b-hf',
    'decapoda-research/llama-13b-hf',
    "decapoda-research/llama-30b-hf",
    "decapoda-research/llama-65b-hf"
]

class HuggingFaceNlpDataLoader(DataLoader):
    def __init__(self, run_config):
        super().__init__(run_config)
        if run_config.model in LLAMA_MODEL_NAMES:
            self._tokenizer = LlamaTokenizer.from_pretrained(run_config.model, padding_side=run_config.padding_side)
            self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        elif run_config.model == 't5-3b':
            self._tokenizer = AutoTokenizer.from_pretrained(
                run_config.model,
                padding_side=run_config.padding_side,
                model_max_length = 8200)
            self._tokenizer.pad_token = self._tokenizer.eos_token
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(run_config.model, padding_side=run_config.padding_side)
            self._tokenizer.pad_token = self._tokenizer.eos_token

        input_sentences = INPUTS
        if run_config.batch_size > len(INPUTS):
            input_sentences *= math.ceil(run_config.batch_size / len(input_sentences))

        self._loaded_data_x = []
        one_data_item = None
        meet_seq_len_inputs = []
        if run_config.seq_len > 1:
            for sen in input_sentences:
                token = self._tokenizer(sen, return_tensors="pt")
                meet_seq_len_input = {}
                for k in token:
                    rp_times = math.ceil(run_config.seq_len / len(token[k][0]))
                    new_token = token[k][0].repeat((1, rp_times))
                    new_token = torch.tensor(new_token[0].numpy()[:run_config.seq_len]).reshape(1, run_config.seq_len)
                    meet_seq_len_input[k] = new_token

                meet_seq_len_inputs.append(meet_seq_len_input)

            if len(meet_seq_len_inputs) >= run_config.total_sample_count:
                self._loaded_data_x = meet_seq_len_inputs[:run_config.total_sample_count]
            else:
                for _ in range(run_config.total_sample_count):
                    self._loaded_data_x.append(meet_seq_len_inputs[np.random.randint(len(meet_seq_len_inputs))])
        else:
            # Since seq_len must be > 1, so this is just for padding
            # test internally, not exposed for user to use. Need by pass
            # the seq_len check in run_config
            tokens = self._tokenizer.batch_encode_plus(
                input_sentences[:run_config.batch_size],
                return_tensors="pt",
                padding=True)
            one_data_item = tokens

            for _ in range(run_config.total_sample_count):
                self._loaded_data_x.append(one_data_item)

        for k in self._loaded_data_x[0]:
            logger.info(F"Loaded data shape: {k}={self._loaded_data_x[0][k].shape}")

        if run_config.verbose:
            logger.info(F"Loaded data: {meet_seq_len_inputs}")

    def get_batch_items(self, batch_size):
        return super().get_batch_items(batch_size)

    def make_batch(self, data_array, selected_indeices):
        batch_inputs = {}
        for k in data_array[0]:
            batch_inputs[k] = torch.cat([data_array[i][k] for i in selected_indeices], dim=0)
        return batch_inputs

    def post_process(self, results):
        if self._run_config.verbose:
            logger.info(F"Output data shape: {results.shape}")
            decoded = self._tokenizer.batch_decode(results, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            logger.info(F"Output decoded data: {decoded}")


class BenchmarkDataLoader(HuggingFaceNlpDataLoader):
    def __init__(self, run_config):
        super(BenchmarkDataLoader, self).__init__(run_config)
