import sys
import os
import time
import subprocess
sys.path.append('/workspace/external')
import global_vars


subprocess.run("rm -rf my_disk*", shell=True)

NUM_GPUS = 1
BATCH_SIZE = 20
NUM_ITERATIONS = 100000
TOTAL_SESSION_IN_BATCH_SIZE = 3
OUTPUT_TOKENS = 100
LEN_INPUT_IN_WORDS = 1000
LEN_WORD = 6

import random
import string

def get_constant():
    return ["Hello my name is Roman I am 41 years old, I work in Nvidia I love my work and I love SPDK, my dream is to write my own fsde that uses spdk glorious API " * 400]

def get_rand_req(n,l):
    return " ".join([''.join(random.choices(string.ascii_letters, k=l)) for _ in range(n)])


sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
os.environ["LMCACHE_CONFIG_FILE"] = os.path.abspath("lmcache_config.yaml")

from vllm import LLM,  SamplingParams
from vllm.config import KVTransferConfig
from lmcache.experimental.config import LMCacheEngineConfig

batch_chunks_dict = {}

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



prompts = [[get_rand_req(LEN_INPUT_IN_WORDS, LEN_WORD) for _ in range(BATCH_SIZE)] for k in range(TOTAL_SESSION_IN_BATCH_SIZE)]

prefetch_dict = []
for i in range(TOTAL_SESSION_IN_BATCH_SIZE):
    global_vars.chunk_hashes_of_curr_batch = []
    outputs = llm.generate(prompts[i], sampling_params=sampling_params)
    prefetch_dict.append(global_vars.chunk_hashes_of_curr_batch)

start_time = time.perf_counter()
prev_futures = [] #[(future, key), (), ()]
for i in range(NUM_ITERATIONS):
    
    print (f"iteration {i}")  
    global_vars.backend.free_prefetched()
    d = {}
    for (future, key) in prev_futures:
        
        while not future.done():
            pass
        res = future.result()
        if res:
            d[key] = future.result()
        else:
            print(f"NO MEMORRY FOR PREFETCHING KEY {key} 555555555555555555555555555555555555555555555555555555555555555555555555555555555555")
    global_vars.backend.feed_prefetched(d)
    prev_futures = []
    for key in prefetch_dict[(i + 1) % TOTAL_SESSION_IN_BATCH_SIZE]:
        (future, key) = global_vars.backend.prefetch(key)
        if future:
            prev_futures.append((future, key))
    
    outputs = llm.generate(prompts[i % TOTAL_SESSION_IN_BATCH_SIZE], sampling_params=sampling_params)




end_time = time.perf_counter()
elapsed = end_time - start_time
print(outputs[0].outputs[0].text) 
print("DONE")
tokens_per_second = (NUM_ITERATIONS * OUTPUT_TOKENS * BATCH_SIZE) // elapsed
print (f"num tokens per second is {tokens_per_second}")
