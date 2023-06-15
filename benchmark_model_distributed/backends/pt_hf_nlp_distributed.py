
import torch
import deepspeed

from anubis_logger import logger
from utilities import print_dict
from backend import TorchDistributedBackend

from time import perf_counter, perf_counter_ns

from transformers import AutoModelForCausalLM, T5ForConditionalGeneration
from transformers import StoppingCriteria, StoppingCriteriaList

CUSTOMIZED_CAUSAL_LM_MODELS = {
    "t5-3b": T5ForConditionalGeneration,
}

torch.manual_seed(20231212)

class TokenTimestampRecoder(StoppingCriteria):
    def __init__(self):
        super().__init__()
        self.token = 0
        self.timestamps = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops=[]):
        token_time = perf_counter()
        self.token += 1
        self.timestamps.append(token_time)
        return False

class BenchmarkBackend(TorchDistributedBackend):
    def __init__(self, run_config):
        super(BenchmarkBackend, self).__init__(run_config)
        self._model_name = run_config.model
        self._dtype = getattr(torch, run_config.dtype)
        self._build_generate_kwargs()
        self._token_timestamp_recoder = TokenTimestampRecoder()
        self.token_timestamps = []

        self._device = torch.device(f"cuda:{run_config.local_rank}") if run_config.distributed else torch.device('cuda')
        if run_config.num_threads > 0:
            torch.set_num_threads(run_config.num_threads)

        logger.info(f"pytorch is using device {self._device}")

    def model_info(self):
        return self._model

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch"

    def load_model(self):
        logger.info(f"Loading model {self._model_name}...")
        if self._model_name in CUSTOMIZED_CAUSAL_LM_MODELS:
            self._model = CUSTOMIZED_CAUSAL_LM_MODELS[self._model_name].from_pretrained(self._model_name)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(self._model_name)

        self._model.eval()

        if self._dtype == torch.float16:
            self._model.half()

        if self._run_config.distributed:
            self._model = deepspeed.init_inference(
                self._model,
                dtype=self._dtype,
                mp_size=self._run_config.world_size,
                replace_with_kernel_inject = self._run_config.use_kernel,
                max_tokens = self._run_config.max_tokens)

        self._model.cuda().to(self._device)
        torch.cuda.empty_cache()

    def predict(self, input_tokens):
        # return
        for t in input_tokens:
            input_tokens[t] = input_tokens[t].to(self._device)

        outputs = self._model.generate(
            **input_tokens, 
            **self._generate_kwargs,
            stopping_criteria=StoppingCriteriaList([self._token_timestamp_recoder]),
            )
        return outputs
    
    def predict_with_perf(self, input_tokens):
        self._token_timestamp_recoder.timestamps = []
        res = super().predict_with_perf(input_tokens)
        token_generate_time = [t - self.start_predict_time for t in self._token_timestamp_recoder.timestamps]
        for i in reversed(range(1, len(token_generate_time))):
            token_generate_time[i] = token_generate_time[i] - token_generate_time[i - 1]
        self.token_predict_times.append(token_generate_time)
        
        return res

    def _build_generate_kwargs(self):
        self._generate_kwargs = {}

        self._generate_kwargs["max_new_tokens"] = self._run_config.max_new_tokens if self._run_config.max_new_tokens > 2 else 2

        if self._run_config.use_cache:
            self._generate_kwargs["use_cache"] = True

        if self._run_config.pad_token_id > -1:
            self._generate_kwargs["pad_token_id"] = self._run_config.pad_token_id

        if self._run_config.greedy:
            self._generate_kwargs["do_sample"] = False
            self._generate_kwargs["num_beams"] = 1
        else:
            self._generate_kwargs["do_sample"] = True
            self._generate_kwargs["num_beams"] = self._run_config.num_beams

        print_dict("Huggingface text generation configs used", self._generate_kwargs)
