import sys
import os
import time
import subprocess
import global_vars
import random
import string
import yaml
import pickle
import random


MAX_LOCAL_CPU_SIZE = 800
MAX_LOCAL_DISK_SIZE = 300
LOCAL_CPU = True
LOCAL_DISK = None
CHUNK_SIZE = 32 * 1024
LMCACHE_CHUNK_SIZE =  32 * 1024 

TOKEN_KV_SIZE = 128 * 1024
INPUT_TOKENS = CHUNK_SIZE 
OUTPUT_TOKENS = 1
LEN_WORD = 6
NUM_ITERATIONS = 10
WITH_STORAGE = False
GPU_MEM_UTILIZATION_RATIO = 0.6
GPU_MEM = 80 * 1024 * 1024 * 1024 

sys.path.append('/workspace/external')
subprocess.run("rm -rf /tmp/rai/*", shell=True)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
os.environ["LMCACHE_CONFIG_FILE"] = os.path.abspath("lmcache_config.yaml")
# ANSI escape codes for colors
BRIGHT_GREEN = "\033[92m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_YELLOW = "\033[93m"
RESET = "\033[0m"


def count_tokens(text, tokenizer):
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_rand_req(n,l, tokenizer):
    print("Starting creating rand request")
    req = "hi"
    while count_tokens(req, tokenizer) < (n - 1000):
        req += " ".join([''.join(random.choices(string.ascii_letters, k=l)) for _ in range(240)])
    while count_tokens(req, tokenizer) < n:
        req += "hi"
    if count_tokens(req, tokenizer) != n:
        raise Exception("couldnt generate good input sequence")
    print("Finished creating rand request")
    return req








# Read the existing config
with open("lmcache_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Modify values
config["is_rai"] = False
config["chunk_size"] = LMCACHE_CHUNK_SIZE
config["local_cpu"] = LOCAL_CPU
config["local_disk"] = LOCAL_DISK
config["max_local_disk_size"] = MAX_LOCAL_DISK_SIZE
config["max_local_cpu_size"] = MAX_LOCAL_CPU_SIZE

# Write back to the file
with open("lmcache_config.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False)

from itertools import product
from vllm import LLM,  SamplingParams
from vllm.config import KVTransferConfig
from lmcache.experimental.config import LMCacheEngineConfig

for (is_kv_in_hbm, tp, BATCH_SIZE) in product([True, False], [1,2,4,8], [1,2,4,8]):
    # Create filename based on the combination
    filename = f"output_{is_kv_in_hbm}_{tp}_{BATCH_SIZE}.txt"
    
    kv_cache_size = TOKEN_KV_SIZE * (INPUT_TOKENS + OUTPUT_TOKENS) 
    total_hbm_memory = tp * GPU_MEM * GPU_MEM_UTILIZATION_RATIO 

    if is_kv_in_hbm:
        total_sessions = 4
        if total_sessions * kv_cache_size >= total_hbm_memory / 1.5:
            print ("BAD SIZING")
            exit(1)
    else:
        total_sessions = int((total_hbm_memory / kv_cache_size) * 1.5)
        if total_sessions * kv_cache_size >= (MAX_LOCAL_CPU_SIZE * 1024 * 1024 * 1024) / 1.5:
            print ("BAD SIZING")
            exit(1)        
    TOTAL_SESSION_IN_BATCH_SIZE = total_sessions // BATCH_SIZE

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
        gpu_memory_utilization = GPU_MEM_UTILIZATION_RATIO,
        max_model_len=100000,
        tensor_parallel_size = tp,
        enable_prefix_caching=True,
        max_num_batched_tokens=CHUNK_SIZE,
    )

    tokenizer = llm.get_tokenizer()


    # Check if prompts file exists, load it if it does, otherwise generate and save
    prompts_filename = f"prompts.{INPUT_TOKENS}.{LEN_WORD}.{BATCH_SIZE}.{TOTAL_SESSION_IN_BATCH_SIZE}"

    if os.path.exists(prompts_filename):
        print(f"Loading prompts from {prompts_filename}")
        with open(prompts_filename, 'rb') as f:
            prompts = pickle.load(f)
    else:
        print(f"Generating new prompts and saving to {prompts_filename}")
        prompts = [[get_rand_req(INPUT_TOKENS, LEN_WORD, tokenizer) for _ in range(BATCH_SIZE)] for k in range(TOTAL_SESSION_IN_BATCH_SIZE)]
        with open(prompts_filename, 'wb') as f:
            pickle.dump(prompts, f)

    for batch in prompts:
        for p in batch:
            print(f"{BRIGHT_GREEN}prompt has {count_tokens(p, tokenizer)} tokens{RESET}")
    #remove later:
    sys.stderr = sys.__stderr__


    print(f"{BRIGHT_GREEN}=========== SIMULATION PARAMETERS ==========={RESET}", file=sys.stderr)
    print(f"{BRIGHT_BLUE}cache: {BRIGHT_GREEN}with{BRIGHT_YELLOW} KV cache{RESET}", file=sys.stderr)
    print(f"{BRIGHT_BLUE}simulation model: {BRIGHT_YELLOW}Llama-3.1-8B-Instruct{RESET}", file=sys.stderr)
    print(f"{BRIGHT_BLUE}num GPUs: {BRIGHT_YELLOW}{tp}{RESET}", file=sys.stderr)
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

    # Access model architecture
    model_config = llm.llm_engine.model_config

    print(f"{BRIGHT_YELLOW}model_config: {model_config}{RESET}")

    num_layers = model_config.hf_config.num_hidden_layers
    num_heads = model_config.hf_config.num_attention_heads
    hidden_size = model_config.hf_config.hidden_size
    head_dim = hidden_size // num_heads

    print(f"{BRIGHT_GREEN}Model architecture:{RESET}")
    print(f"{BRIGHT_BLUE}Number of layers: {BRIGHT_YELLOW}{num_layers}{RESET}")
    print(f"{BRIGHT_BLUE}Number of attention heads: {BRIGHT_YELLOW}{num_heads}{RESET}")
    print(f"{BRIGHT_BLUE}Head dimension: {BRIGHT_YELLOW}{head_dim}{RESET}")
    print(f"{BRIGHT_BLUE}Hidden size: {BRIGHT_YELLOW}{hidden_size}{RESET}")


    print(f"FILL THE CACHE STAGE")


    prefetch_dict = []
    for i in range(TOTAL_SESSION_IN_BATCH_SIZE):
        if WITH_STORAGE:
            global_vars.chunk_hashes_of_curr_batch = []
        outputs = llm.generate(prompts[i], sampling_params=sampling_params)

        if WITH_STORAGE:
            prefetch_dict.append(global_vars.chunk_hashes_of_curr_batch)

    print(f"{BRIGHT_GREEN}=========== PREFETCHING DONE ==========={RESET}", file=sys.stderr)
    # exit()

    print(f"DONE WITH FILL STAGE")

    start_time = time.perf_counter()


    if WITH_STORAGE:
        prev_futures = [] #[(future, key), (), ()]
        for i in range(NUM_ITERATIONS):
            x = (i-1)*(BATCH_SIZE*OUTPUT_TOKENS) 
            y = time.perf_counter() - start_time
            print (f"iteration {i} and time elapsed {y} and the rate is {x / y}", file=sys.stderr)  
            global_vars.backend.free_prefetched()
            d = {}
            for (future, key) in prev_futures:
                while not future.done():
                    print(f"waiting for future in iteration {i} for key {key.to_string()}", file=sys.stderr)
                    pass
                res = future.result()
                if res:
                    d[key] = future.result()
                else:
                    print(f"NO MEMORRY FOR PREFETCHING KEY {key} 555555555555555555555555555555555555555555555555555555555555555555555555555555555555", file=sys.stderr)
            global_vars.backend.feed_prefetched(d)
            prev_futures = []
            for key in prefetch_dict[(i + 1) % TOTAL_SESSION_IN_BATCH_SIZE]:
                (future, key) = global_vars.backend.prefetch(key)
                if future:
                    prev_futures.append((future, key))
            outputs = llm.generate(prompts[i % TOTAL_SESSION_IN_BATCH_SIZE], sampling_params=sampling_params)
    else:
        for i in range(NUM_ITERATIONS):
            x = (i-1)*(BATCH_SIZE*OUTPUT_TOKENS) 
            y = time.perf_counter() - start_time
            print (f"iteration {i} and time elapsed {y} and the rate is {x / y}", file=sys.stderr)  
            outputs = llm.generate(prompts[i % TOTAL_SESSION_IN_BATCH_SIZE], sampling_params=sampling_params)
            metrics = outputs[0].metrics
            # TTFT in seconds
            ttft = metrics.first_token_time - metrics.first_scheduled_time
            print(f"XXXXXXXXXXXXXXXXXXYYYYYYYYYYYYYYYYYYYYYYYYYYYY       Time to first token: {ttft:.3f} seconds")




    end_time = time.perf_counter()
    elapsed = end_time - start_time
    tokens_per_second = (NUM_ITERATIONS *  OUTPUT_TOKENS * BATCH_SIZE) / elapsed
    
    # Write tokens_per_second to the file
    with open(filename, 'w') as f:
        f.write(f"tokens_per_second = {tokens_per_second:.2f}\n")

    sampling_params2 = SamplingParams(
        max_tokens=50,
        min_tokens=5,
        ignore_eos=True
    )
    print(f"=========== TESTING JOKE ==========={RESET}", file=sys.stderr)
    joke_outputs = llm.generate("Tell me a joke", sampling_params=sampling_params2)
    print(f"joke: {joke_outputs[0].outputs[0].text}", file=sys.stderr)


    

    print(f"{BRIGHT_GREEN}=========== SIMULATION RESULTS ==========={RESET}", file=sys.stderr)
    print(f"{BRIGHT_BLUE}New tokens rate is {BRIGHT_YELLOW}{tokens_per_second:.2f}{RESET}", file=sys.stderr)
    print(f"{BRIGHT_BLUE}Elapsed time is {BRIGHT_YELLOW}{elapsed:.2f}{RESET}", file=sys.stderr)




subprocess.run("pkill -9 python3", shell=True)