import sys
import os
import time
import subprocess
import global_vars
import random
import string

sys.path.append('/workspace/external')
subprocess.run("rm -rf /tmp/rai/*", shell=True)


NUM_GPUS = 4
BATCH_SIZE = 1
NUM_ITERATIONS = 30
TOTAL_SESSION_IN_BATCH_SIZE = 5
OUTPUT_TOKENS = 1
LEN_INPUT_IN_WORDS = 20000
LEN_WORD = 6

# NUM_GPUS = 1
# BATCH_SIZE = 1
# NUM_ITERATIONS = 20
# TOTAL_SESSION_IN_BATCH_SIZE = 3
# OUTPUT_TOKENS = 100
# LEN_INPUT_IN_WORDS = 1000
# LEN_WORD = 6


# ANSI escape codes for colors
BRIGHT_GREEN = "\033[92m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_YELLOW = "\033[93m"
RESET = "\033[0m"



def get_rand_req(n,l):
    return " ".join([''.join(random.choices(string.ascii_letters, k=l)) for _ in range(n)])


# #remove later:
# sys.stderr = open('log.txt', 'w')

import random



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
    gpu_memory_utilization = 0.6,
    max_model_len=100000,
    tensor_parallel_size = NUM_GPUS,
    enable_prefix_caching=True,
)



prompts = [[get_rand_req(LEN_INPUT_IN_WORDS, LEN_WORD) for _ in range(BATCH_SIZE)] for k in range(TOTAL_SESSION_IN_BATCH_SIZE)]
# prompts = generate_queries(num_queries=TOTAL_SESSION_IN_BATCH_SIZE * BATCH_SIZE, words_per_query=LEN_INPUT_IN_WORDS)


#remove later:
sys.stderr = sys.__stderr__


print(f"{BRIGHT_GREEN}=========== SIMULATION PARAMETERS ==========={RESET}", file=sys.stderr)
print(f"{BRIGHT_BLUE}cache: {BRIGHT_GREEN}with{BRIGHT_YELLOW} KV cache{RESET}", file=sys.stderr)
print(f"{BRIGHT_BLUE}simulation model: {BRIGHT_YELLOW}Llama-3.1-8B-Instruct{RESET}", file=sys.stderr)
print(f"{BRIGHT_BLUE}num GPUs: {BRIGHT_YELLOW}{NUM_GPUS}{RESET}", file=sys.stderr)
print(f"{BRIGHT_BLUE}batch size: {BRIGHT_YELLOW}{BATCH_SIZE}{RESET}", file=sys.stderr)
print(f"{BRIGHT_BLUE}number of sessions: {BRIGHT_YELLOW}{BATCH_SIZE * TOTAL_SESSION_IN_BATCH_SIZE}{RESET}", file=sys.stderr)
print(f"{BRIGHT_BLUE}queue size: {BRIGHT_YELLOW}{BATCH_SIZE}{RESET}", file=sys.stderr)

print(f"{BRIGHT_GREEN}=========== LOADING MODEL ==========={RESET}", file=sys.stderr)
for i in range(16):
    if i == 15:
        print(".", flush=True, file=sys.stderr)
    else:
        print(".", end="", flush=True, file=sys.stderr)
    time.sleep(0.1)




print(f"{BRIGHT_GREEN}=========== SIMULATION STARTING ==========={RESET}", file=sys.stderr)


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
            print(f"waiting for future in iteration {i} for key {key.to_string()}")
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
# print("DONE")
tokens_per_second = (NUM_ITERATIONS *  OUTPUT_TOKENS * BATCH_SIZE) / elapsed
# 


# os.system('clear')




print(f"{BRIGHT_GREEN}=========== SIMULATION RESULTS ==========={RESET}", file=sys.stderr)

print(f"{BRIGHT_BLUE}New tokens rate is {BRIGHT_YELLOW}{tokens_per_second:.2f}{RESET}", file=sys.stderr)
print(f"{BRIGHT_BLUE}Elapsed time is {BRIGHT_YELLOW}{elapsed:.2f}{RESET}", file=sys.stderr)


exit()

