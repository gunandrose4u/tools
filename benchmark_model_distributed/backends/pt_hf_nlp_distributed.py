
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
        self.timestamps = []

    # when framework is calling this function, it means the token is generated, but the tensor is on GPU
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops=[]):
        token_time = perf_counter()
        self.timestamps.append(token_time)
        return False

class BenchmarkBackend(TorchDistributedBackend):
    def __init__(self, run_config):
        super(BenchmarkBackend, self).__init__(run_config)
        self._model_name = run_config.model
        self._token_timestamp_recoder = TokenTimestampRecoder()
        self.token_timestamps = []

        self._device = torch.device(f"cuda:{run_config.local_rank}") if run_config.distributed else torch.device('cuda:0')
        if run_config.num_threads > 0:
            torch.set_num_threads(run_config.num_threads)

        if run_config.dtype == 'bfloat16':
            self._amp_enabled = True
            self._dtype = torch.bfloat16
        elif run_config.dtype == 'float16':
            self._amp_enabled = True
            self._dtype = torch.float16
        else:
            self._amp_enabled = False
            self._dtype = torch.float32

        self._build_generate_kwargs()

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
            self._model = CUSTOMIZED_CAUSAL_LM_MODELS[self._model_name].from_pretrained(
                self._model_name,
                torch_dtype=self._dtype
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=self._dtype
            )
        self._model.eval()

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
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=self._amp_enabled, dtype=self._dtype if self._amp_enabled else None):
            for t in input_tokens:
                input_tokens[t] = input_tokens[t].to(self._device)

            outputs = self._model.generate(
                **input_tokens,
                **self._generate_kwargs,
                )

            return outputs

    def predict_with_perf(self, input_tokens):
        self._token_timestamp_recoder.timestamps.clear()
        res = super().predict_with_perf(input_tokens)
        if self._run_config.token_record:
            token_generate_time = [t - self.start_predict_time for t in self._token_timestamp_recoder.timestamps]
            for i in reversed(range(1, len(token_generate_time))):
                token_generate_time[i] = token_generate_time[i] - token_generate_time[i - 1]
            self.token_predict_times.append(token_generate_time)

        return res

    def _build_generate_kwargs(self):
        self._generate_kwargs = {}

        self._generate_kwargs["max_new_tokens"] = self._run_config.max_new_tokens
        self._generate_kwargs["use_cache"] = self._run_config.use_cache

        if self._run_config.pad_token_id > -1:
            self._generate_kwargs["pad_token_id"] = self._run_config.pad_token_id

        if self._run_config.greedy:
            self._generate_kwargs["do_sample"] = False
            self._generate_kwargs["num_beams"] = 1
        else:
            self._generate_kwargs["do_sample"] = self._run_config.do_sample
            self._generate_kwargs["num_beams"] = self._run_config.num_beams

        if self._run_config.token_record:
            self._generate_kwargs["stopping_criteria"] = StoppingCriteriaList([self._token_timestamp_recoder])

        print_dict("Huggingface text generation configs used", self._generate_kwargs)
