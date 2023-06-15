import math

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
LLAMA_MODEL_NAMES = ['decapoda-research/llama-7b-hf', 'decapoda-research/llama-13b-hf']

class BenchmarkDataLoader(DataLoader):
    def __init__(self, run_config):
        super().__init__(run_config)
        if run_config.model in LLAMA_MODEL_NAMES:
            self._tokenizer = LlamaTokenizer.from_pretrained(run_config.model, padding_side=run_config.padding_side)
            self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(run_config.model, padding_side=run_config.padding_side)
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._loaded_data_x = []
        if run_config.seq_len > 1:
            for sentence in INPUTS:
                rough_words = sentence.split(' ')
                rough_words = rough_words * math.ceil(run_config.seq_len / len(rough_words))
                new_sentence = ' '.join(rough_words[:run_config.seq_len])
                self._loaded_data_x.append(new_sentence)
        else:
            self._loaded_data_x = INPUTS

        if run_config.verbose:
            logger.info("Loaded data: ")
            for d in self._loaded_data_x:
                logger.info(d)
                logger.info(len(d.split(' ')))


    def get_batch_items(self, batch_size=1):
        input_sentences = self._loaded_data_x
        if batch_size > len(INPUTS):
            input_sentences *= math.ceil(batch_size / len(input_sentences))

        return  self._tokenizer.batch_encode_plus(input_sentences[:batch_size], return_tensors="pt", padding=True)

    def post_process(self, results):
        if self._run_config.verbose:
            decoded = self._tokenizer.batch_decode(results, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i in range(len(decoded)):
                logger.info(decoded[i])
                logger.info(len(decoded[i]))
