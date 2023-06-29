
from benchmarkers.direct_benchmarker import DirectBenchmarker
# from benchmarkers.mlperf.mlperf_benchmarker import MlPerfBenchmarker
from benchmarkers.nlp_generative_benchmarker import NlpGenerativeBenchmarker


# Will release direct and mlperf benchmarker in the future
# if you need to use them, please uncomment the following lines
BENCHMARKER_MAPPING = {
    # "direct": DirectBenchmarker,
    # "mlperf": MlPerfBenchmarker,
    "nlp_generative": NlpGenerativeBenchmarker,
}

PT_HF_NLP_GENERATIVE_BENCHMARK_CONFIG = ["pt_hf_nlp_distributed", "pt_hf_nlp", "nlp_generative"]
PT_MS_BLOOM_GENERATIVE_BENCHMARK_CONFIG = ["pt_ms_bloom_distributed", "pt_hf_nlp", "nlp_generative"]

SUPPORTED_MODELS = {
    "facebook/opt-1.3b": PT_HF_NLP_GENERATIVE_BENCHMARK_CONFIG,
    "t5-3b": PT_HF_NLP_GENERATIVE_BENCHMARK_CONFIG,
    "EleutherAI/gpt-j-6B": PT_HF_NLP_GENERATIVE_BENCHMARK_CONFIG,
    "decapoda-research/llama-7b-hf": PT_HF_NLP_GENERATIVE_BENCHMARK_CONFIG,
    "decapoda-research/llama-13b-hf": PT_HF_NLP_GENERATIVE_BENCHMARK_CONFIG,
    "decapoda-research/llama-30b-hf": PT_HF_NLP_GENERATIVE_BENCHMARK_CONFIG,
    "decapoda-research/llama-65b-hf": PT_HF_NLP_GENERATIVE_BENCHMARK_CONFIG,
    "bigscience/bloom-7b1": PT_HF_NLP_GENERATIVE_BENCHMARK_CONFIG,
    "bigscience/bloom": PT_HF_NLP_GENERATIVE_BENCHMARK_CONFIG,
    "microsoft/bloom-deepspeed-inference-fp16": PT_MS_BLOOM_GENERATIVE_BENCHMARK_CONFIG,
}