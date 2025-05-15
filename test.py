import sys
import os
import time
import subprocess

subprocess.run("rm -rf my_disk*", shell=True)

NUM_GPUS = 1
BATCH_SIZE = 20
NUM_ITERATIONS = 100000
TOTAL_SCALE = 60
OUTPUT_TOKENS = 100
LEN_INPUT_IN_WORDS = 1000
LEN_WORD = 6

import random
import string

def get_constant():
    return ["Hello my name is Roman I am 41 years old, I work in Nvidia I love my work and I love SPDK, my dream is to write my own fsde that uses spdk glorious API " * 400]

def get_rand_req(n,l):
    return " ".join([''.join(random.choices(string.ascii_letters, k=l)) for _ in range(n)])


import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

#os.environ["LMCACHE_LOCAL_CPU"] = "True"
#os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5.0"  # Example: 5 GB
#os.environ["LMCACHE_CHUNK_SIZE"] = "16"
os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"


#os.environ["LMCACHE_LOCAL_CPU"] = "False"
#os.environ["LMCACHE_LOCAL_DISK"] = "False"
os.environ["LMCACHE_CONFIG_FILE"] = os.path.abspath("lmcache_config.yaml")

from vllm import LLM,  SamplingParams
from vllm.config import KVTransferConfig
from lmcache.experimental.config import LMCacheEngineConfig


lmcache_config = LMCacheEngineConfig.from_file("lmcache_config.yaml")



sampling_params = SamplingParams(
    max_tokens=OUTPUT_TOKENS,
    min_tokens=OUTPUT_TOKENS,
    ignore_eos=True
)


llm = LLM(
    model="/workspace/llm_models/llama-3.1-model/Llama-3.1-8B-Instruct",
    kv_transfer_config=KVTransferConfig(
        kv_connector="LMCacheConnector",
        kv_role="kv_both",
        lmcache_config=lmcache_config
    ),
    gpu_memory_utilization = 0.25,
    max_model_len=8192,
    tensor_parallel_size = NUM_GPUS
    #what about pipeline parralelism
)
sentences = [get_rand_req(LEN_INPUT_IN_WORDS, LEN_WORD) for _ in range(TOTAL_SCALE)]


start_time = time.perf_counter()
for i in range(NUM_ITERATIONS):
    print (f"iteration {i}")
    prompts = [sentences[j % len(sentences)] for j in range(i * BATCH_SIZE, i * BATCH_SIZE + BATCH_SIZE)]
    outputs = llm.generate(prompts, sampling_params=sampling_params)
end_time = time.perf_counter()
elapsed = end_time - start_time
print(outputs[0].outputs[0].text) 
print("DONE")
tokens_per_second = (NUM_ITERATIONS * OUTPUT_TOKENS * BATCH_SIZE) // elapsed
print (f"num tokens per second is {tokens_per_second}")
