
from benchmarkers.direct_benchmarker import DirectBenchmarker
from benchmarkers.nlp_generative_benchmarker import NlpGenerativeBenchmarker

BENCHMARKER_MAPPING = {
    "direct": DirectBenchmarker,
    "nlp_generative": NlpGenerativeBenchmarker,
}

PT_HF_NLP_GENERATIVE_BENCHMARK_CONFIG = ["pt_hf_nlp_distributed", "pt_hf_nlp", "nlp_generative"]

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
}