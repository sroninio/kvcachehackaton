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
import datetime

# ANSI escape codes for colors
BRIGHT_GREEN = "\033[92m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_RED = "\033[91m"
BOLD_RED = "\033[1;91m"
RESET = "\033[0m"

class Statisics:
    def __init__(self, test):
        self.reset()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp_str = timestamp.replace(" ", "_").replace(":", "-")
        self.filename = f"statistics_{timestamp_str}_tp_{test.TP}_chunk{test.CHUNK_SIZE}_input{test.INPUT_TOKENS}_output{test.OUTPUT_TOKENS}_sessions{test.SESSIONS}"

    def reset(self):
        self.curr_disk_inflights = 0
        self.curr_llm_inflights = 0
        self.avg_disk_inflights = (0, 0) # (samples, avg_value)
        self.avg_llm_inflights = (0, 0) # (samples, avg_value) 

    def update(self):
        # Update disk inflights average
        samples_disk, avg_disk = self.avg_disk_inflights
        samples_disk += 1
        new_avg_disk = (avg_disk * (samples_disk - 1) + self.curr_disk_inflights) / samples_disk
        self.avg_disk_inflights = (samples_disk, new_avg_disk)
        
        # Update LLM inflights average
        samples_llm, avg_llm = self.avg_llm_inflights
        samples_llm += 1
        new_avg_llm = (avg_llm * (samples_llm - 1) + self.curr_llm_inflights) / samples_llm
        self.avg_llm_inflights = (samples_llm, new_avg_llm)
        
    
         
class VLLM_BENCHMARK:
    def __init__(self, 
                 max_local_cpu_size=20,
                 max_local_disk_size=300,
                 local_cpu=True,
                 #local_disk = "file:///tmp/abc/",
                 local_disk = "file:///tmp/abc/",
                 #chunk_size= 3 * 32 * 1024 + 1024,
                 chunk_size= 32 * 1024,
                 lmcache_chunk_size=32 * 1024,
                 input_tokens = 32 * 1024,
                 output_tokens = 1024,
                 len_word=6,
                 always_hit_in_cpu = False,
                 gpu_mem_utilization_ratio=0.6,
                 gpu_mem=80 * 1024 * 1024 * 1024,
                 tp=2,
                 sessions=50):
        # Configuration constants
        self.MAX_LOCAL_CPU_SIZE = max_local_cpu_size
        self.MAX_LOCAL_DISK_SIZE = max_local_disk_size
        self.LOCAL_CPU = local_cpu
        self.LOCAL_DISK = local_disk
        self.CHUNK_SIZE = chunk_size
        self.LMCACHE_CHUNK_SIZE = lmcache_chunk_size
        self.ALWAYS_HIT_IN_CPU = always_hit_in_cpu
        self.INPUT_TOKENS = input_tokens if input_tokens is not None else chunk_size
        self.OUTPUT_TOKENS = output_tokens
        self.LEN_WORD = len_word
        self.GPU_MEM_UTILIZATION_RATIO = gpu_mem_utilization_ratio
        self.GPU_MEM = gpu_mem
        self.TP = tp
        self.SESSIONS = sessions
        self.statistics = Statisics(self)
        self.terminate = False
        self.sampling_params, self.engine = self.create_engine(self.global_configure()) 


    def count_tokens(self, text, tokenizer):
        tokens = tokenizer.encode(text)
        return len(tokens)

    def get_rand_req(self, n, l, tokenizer):
        while True:
            word = ''.join(random.choices(string.ascii_lowercase, k=6))
            req = word + " " + "hi " * (n-10)
            while self.count_tokens(req, tokenizer) < n:
                req += "hi"
            if self.count_tokens(req, tokenizer) == n:
                print("Finished creating rand request")
                return req

    def get_rand_req2(self, n, l, tokenizer):
        print("Starting creating rand request")
        req = "hi"
        while self.count_tokens(req, tokenizer) < (n - 1000):
            req += " ".join([''.join(random.choices(string.ascii_letters, k=l)) for _ in range(240)])
        while self.count_tokens(req, tokenizer) < n:
            req += "hi"
        if self.count_tokens(req, tokenizer) != n:
            raise Exception("couldnt generate good input sequence")
        print("Finished creating rand request")
        return req

    def global_configure(self):
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
        config["chunk_size"] = self.LMCACHE_CHUNK_SIZE
        config["always_hit_in_cpu"] = self.ALWAYS_HIT_IN_CPU
        config["local_cpu"] = self.LOCAL_CPU
        config["local_disk"] = self.LOCAL_DISK
        config["max_local_disk_size"] = self.MAX_LOCAL_DISK_SIZE
        config["max_local_cpu_size"] = self.MAX_LOCAL_CPU_SIZE

        # Write back to the file
        with open("lmcache_config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        lmcache_config = LMCacheEngineConfig.from_file("lmcache_config.yaml")
        return lmcache_config

    def create_engine(self, lmcache_config):
        sampling_params = SamplingParams(
            max_tokens=self.OUTPUT_TOKENS,
            min_tokens=self.OUTPUT_TOKENS,
            ignore_eos=True
        )

        engine_args = AsyncEngineArgs(
            model="/workspace/llm_models/llama-3.1-model/Llama-3.1-8B-Instruct",
            #model = "/workspace/llm_models/llama-3.3-models/Llama-3.3-70B-Instruct",
            kv_transfer_config=KVTransferConfig(
                kv_connector="LMCacheConnector",
                kv_role="kv_both",
                lmcache_config=lmcache_config
            ),
            gpu_memory_utilization=self.GPU_MEM_UTILIZATION_RATIO,
            max_model_len=50000,
            tensor_parallel_size=self.TP,
            enable_prefix_caching=True,
            max_num_batched_tokens=self.CHUNK_SIZE,
        )

        engine = AsyncLLMEngine.from_engine_args(
            engine_args, 
            usage_context=UsageContext.LLM_CLASS
        ) 
        return sampling_params, engine

    def create_prompts(self):
        # Get tokenizer
        tokenizer = self.engine.engine.get_tokenizer()

        # Get prompts
        prompts_filename = f"prompts_async.{self.INPUT_TOKENS}.{self.LEN_WORD}.{self.SESSIONS}"
        if os.path.exists(prompts_filename):
            print(f"Loading prompts from {prompts_filename}")
            with open(prompts_filename, 'rb') as f:
                prompts = pickle.load(f)
        else:
            print(f"Generating new prompts and saving to {prompts_filename}")
            prompts = [{"req" : self.get_rand_req(self.INPUT_TOKENS, self.LEN_WORD, tokenizer), "keys" : []} for _ in range(self.SESSIONS)]
            with open(prompts_filename, 'wb') as f:
                pickle.dump(prompts, f)
        return prompts

    async def execute_single_request_in_llm(self,  req,  indx):
        async for output in self.engine.generate(req, sampling_params=self.sampling_params, request_id=indx):
            pass 

    async def enter_new_request(self, p, indx):
        import time
        start_time = time.time()
        if global_vars.backend:
            self.statistics.curr_disk_inflights += 1
            for i, mem_obj in enumerate(await global_vars.backend.prefetch_async(p['keys'])):
                if mem_obj:
                    global_vars.backend.add_to_prefetched(p['keys'][i], mem_obj) 
            self.statistics.curr_disk_inflights -= 1
        self.statistics.curr_llm_inflights += 1
        
        print(f"{BOLD_RED}STARTING ASYNC EXECUTION INDX {indx}{RESET}")
        await self.execute_single_request_in_llm(p["req"], indx)
        print(f"{BOLD_RED}FINISHING ASYNC EXECUTION INDX {indx} {RESET}")

        if global_vars.backend:
            for key in p['keys']:
                global_vars.backend.remove_from_prefteched(key) 
        self.statistics.curr_llm_inflights -= 1
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"enter_new_request (index {indx}) took {execution_time:.4f} seconds")

    async def add_hash_keys_to_prompts(self, prompts):
        #get the keys for each prompt
        print (f"GETTING KEYS OF EACH PROMPT")
        for indx, p in enumerate(prompts):
            global_vars.chunk_hashes_of_curr_batch = []
            await self.execute_single_request_in_llm(p["req"], indx)
            p["keys"] = global_vars.chunk_hashes_of_curr_batch
        print (f"FINISHED GETING KEYS OF EACH PROMPT")

    async def update_statistics(self):
        await asyncio.sleep(0.1)
        if self.terminate:
            return
        self.statistics.update()
        self.stat_task = asyncio.create_task(self.update_statistics()) 

    def append_statistics_to_file(self, MAX_INFLIGHTS):
        # Extract statistics data
        disk_samples, avg_disk_inflights = self.statistics.avg_disk_inflights
        llm_samples, avg_llm_inflights = self.statistics.avg_llm_inflights
        
        # Create statistics entry
        stats_entry = {
            "configuration": {
                "inflights": MAX_INFLIGHTS,
            },
            "performance": {
                "reqs_per_second": round(self.statistics.reqs_per_second, 4),
                "avg_disk_inflights": round(avg_disk_inflights, 2),
                "avg_llm_inflights": round(avg_llm_inflights, 2),
                "disk_samples": disk_samples,
                "llm_samples": llm_samples
            }
        }
        # Append to statistics file
        with open(self.statistics.filename, "a") as f:
            f.write(f"{stats_entry}\n")
        
    async def run_benchmark(self, MAX_INFLGITHS, NUM_ITERATIONS):
        prompts = self.create_prompts()
        await self.add_hash_keys_to_prompts(prompts)
        
        running, inflights = set(), 0
        start_time = time.time() 
        
        for i in range(1, NUM_ITERATIONS):
            running.add(asyncio.create_task(self.enter_new_request(prompts[i % len(prompts)], i)))
            inflights += 1
            if inflights == MAX_INFLGITHS:
                done, running = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                inflights -= len(done)
        self.terminate = True
        await asyncio.gather(*running)
        
        # Stop timer and calculate metrics
        total_time = time.time() - start_time
        self.statistics.reqs_per_second = NUM_ITERATIONS / total_time 
        
        
        print(f"{BRIGHT_GREEN}Total execution time: {total_time:.2f} seconds{RESET}")
        print(f"{BRIGHT_BLUE}Requests per second: {self.statistics.reqs_per_second:.4f} seconds{RESET}")
        
        # Append statistics to file
        self.append_statistics_to_file(MAX_INFLGITHS)
        
        

async def main():
    benchmark = VLLM_BENCHMARK()
    for inflights in [256]:
        print(f"{BOLD_RED}STARTING ITERATION WITH {inflights} INFLIGHTS {RESET}")
        benchmark.statistics.reset()
        benchmark.terminate = False
        benchmark.stat_task = asyncio.create_task(benchmark.update_statistics()) 
        await benchmark.run_benchmark(inflights, inflights * 5)
        benchmark.terminate = True
        await benchmark.stat_task
    


if __name__ == "__main__":
    asyncio.run(main())
