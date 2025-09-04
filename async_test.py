import sys
import os
import time
import subprocess
from VLLM.vllm import engine
import global_vars
import random
import string
import yaml
import pickle
import random
from itertools import product
from vllm import LLM,  SamplingParams
from vllm.config import KVTransferConfig
from lmcache.experimental.config import LMCacheEngineConfig
import asyncio
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncEngineArgs
from vllm.usage.usage_lib import UsageContext

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
NUM_ITERATIONS = 100
WITH_STORAGE = False
GPU_MEM_UTILIZATION_RATIO = 0.6
GPU_MEM = 80 * 1024 * 1024 * 1024 
TP = 1
INFLIGHTS = 10
SESSIONS = 20

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

def global_configure():
    sys.path.append('/workspace/external')
    subprocess.run("rm -rf /tmp/rai/*", shell=True)
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
    os.environ["LMCACHE_CONFIG_FILE"] = os.path.abspath("lmcache_config.yaml")

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
    lmcache_config = LMCacheEngineConfig.from_file("lmcache_config.yaml")


def create_engine(lmcache_config):
    sampling_params = SamplingParams(
        max_tokens=OUTPUT_TOKENS,
        min_tokens=OUTPUT_TOKENS,
        ignore_eos=True
    )

    engine_args = AsyncEngineArgs(
        model="/workspace/llm_models/llama-3.1-model/Llama-3.1-8B-Instruct",
        kv_transfer_config=KVTransferConfig(
            kv_connector="LMCacheConnector",
            kv_role="kv_both",
            lmcache_config=lmcache_config
        ),
        gpu_memory_utilization=GPU_MEM_UTILIZATION_RATIO,
        max_model_len=100000,
        tensor_parallel_size=TP,
        enable_prefix_caching=True,
        max_num_batched_tokens=CHUNK_SIZE,
    )

    engine = AsyncLLMEngine.from_engine_args(
        engine_args, 
        usage_context=UsageContext.LLM_CLASS
    ) 
    return sampling_params, engine

def create_prompts(engine):
    # Get tokenizer
    tokenizer = engine.engine.get_tokenizer()

    # Get prompts
    prompts_filename = f"prompts_async.{INPUT_TOKENS}.{LEN_WORD}.{SESSIONS}"
    if os.path.exists(prompts_filename):
        print(f"Loading prompts from {prompts_filename}")
        with open(prompts_filename, 'rb') as f:
            prompts = pickle.load(f)
    else:
        print(f"Generating new prompts and saving to {prompts_filename}")
        prompts = [{"req" : get_rand_req(INPUT_TOKENS, LEN_WORD, tokenizer), "keys" : []} for _ in range(SESSIONS)]
        with open(prompts_filename, 'wb') as f:
            pickle.dump(prompts, f)
    return prompts


async def execute_single_request_in_llm(engine, req, sampling_params, indx):
    async for output in engine.generate(req, sampling_params=sampling_params, request_id=indx):
        pass 

async def enter_new_request(engine, p, sampling_params, indx):
    if global_vars.backend:
        for i, mem_obj in enumerate(await global_vars.backend.prefetch_async(p['keys'])):
            if mem_obj:
                global_vars.backend.add_to_prefetched(p['keys'][i], mem_obj) 
    await execute_single_request_in_llm(engine, p["req"], sampling_params, indx)
    if global_vars.backend:
        for key in p['keys']:
            global_vars.backend.remove_from_prefteched(key) 

    
async def add_hash_keys_to_prompts(engine, prompts, sampling_params):
    #get the keys for each prompt
    print (f"GETTING KEYS OF EACH PROMPT")
    for indx, p in enumerate(prompts):
        global_vars.chunk_hashes_of_curr_batch = []
        await execute_single_request_in_llm(engine, p["req"], sampling_params, indx)
        p["keys"] = global_vars.chunk_hashes_of_curr_batch
    print (f"FINISHED GETING KEYS OF EACH PROMPT")


 
async def main():
    lmcache_config = global_configure()
    sampling_params, engine = create_engine(lmcache_config) 
    prompts = create_prompts(engine)
    await add_hash_keys_to_prompts(engine, prompts, sampling_params)
    running, inflights = set(), 0
    start_time = None
    
    for i in range(NUM_ITERATIONS):
        # Start timer after N/2 iterations
        if i == NUM_ITERATIONS // 2:
            start_time = time.time()
            print(f"{BRIGHT_YELLOW}Starting timer at iteration {i}{RESET}")
            
        running.add(asyncio.create_task(enter_new_request(engine, prompts[i % len(prompts)], sampling_params, i)))
        inflights += 1
        if inflights == INFLIGHTS:
            done, running = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            inflights -= len(done)
    await asyncio.gather(*running)
    
    # Stop timer and calculate metrics
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_iteration = total_time / (NUM_ITERATIONS // 2)
    
    print(f"{BRIGHT_GREEN}Total execution time: {total_time:.2f} seconds{RESET}")
    print(f"{BRIGHT_BLUE}Average time per iteration: {avg_time_per_iteration:.4f} seconds{RESET}")
    print(f"{BRIGHT_YELLOW}Total iterations measured: {NUM_ITERATIONS // 2}{RESET}")

if __name__ == "__main__":
    asyncio.run(main())
