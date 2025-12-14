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
from lmcache.logging import log_to_pid_file
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
        self.filename = f"statistics_{timestamp_str}_tp_{test.TP}_chunk{test.CHUNK_SIZE}_input{test.KVC_LEN_TOKENS}_output{test.OUTPUT_TOKENS}_conversations{test.CONVERSATIONS}_steps{test.STEPS}_isl{test.ISL_LEN_TOKENS}"

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
                 max_local_cpu_size=200,
                 max_local_disk_size=300,
                 local_cpu=True,
                 #local_disk = "file:///tmp/abc/",
                 local_disk = None,
                 #chunk_size= 3 * 32 * 1024 + 1024,
                 chunk_size= 32 * 1024,
                 lmcache_chunk_size=32 * 1024,
                 kvc_len_tokens=32 * 1024,
                 output_tokens=1,
                 num_iterations=50,
                 always_hit_in_cpu = False,
                 gpu_mem_utilization_ratio=0.6,
                 tp=1,
                 conversations=1,
                 steps = 1,
                 isl_len_tokens = 1024,
                 model_path = "/workspace/llm_models/llama-3.1-model/Llama-3.1-8B-Instruct" 
                 ):
        # Configuration constants
        self.MAX_LOCAL_CPU_SIZE = max_local_cpu_size
        self.MAX_LOCAL_DISK_SIZE = max_local_disk_size
        self.LOCAL_CPU = local_cpu
        self.LOCAL_DISK = local_disk
        self.CHUNK_SIZE = chunk_size
        self.LMCACHE_CHUNK_SIZE = lmcache_chunk_size
        self.ALWAYS_HIT_IN_CPU = always_hit_in_cpu
        self.KVC_LEN_TOKENS = kvc_len_tokens if kvc_len_tokens is not None else chunk_size
        self.OUTPUT_TOKENS = output_tokens
        self.NUM_ITERATIONS = num_iterations
        self.GPU_MEM_UTILIZATION_RATIO = gpu_mem_utilization_ratio
        self.TP = tp
        self.CONVERSATIONS = conversations
        self.STEPS = steps
        self.ISL_LEN_TOKENS = isl_len_tokens
        self.MODEL_PATH = model_path

        
 
        self.statistics = Statisics(self)
        self.terminate = False
        self.sampling_params, self.engine = self.create_engine(self.global_configure()) 
        self.vocab_size = self.engine.engine.get_tokenizer().vocab_size 


    def get_rand_req(self, n):
        """Generate a random list of n token IDs from 0 to vocab_size-1."""
        return [random.randint(0, self.vocab_size - 1) for _ in range(n)]


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
            model=self.MODEL_PATH,
            kv_transfer_config=KVTransferConfig(
                kv_connector="LMCacheConnector",
                kv_role="kv_both",
                lmcache_config=lmcache_config
            ),
            gpu_memory_utilization=self.GPU_MEM_UTILIZATION_RATIO,
            max_model_len=100000,
            tensor_parallel_size=self.TP,
            enable_prefix_caching=True,
            max_num_batched_tokens=self.CHUNK_SIZE,
        )

        engine = AsyncLLMEngine.from_engine_args(
            engine_args, 
            usage_context=UsageContext.LLM_CLASS
        ) 
        
        return sampling_params, engine

    def create_isls(self, num_isls):
        return [self.get_rand_req(self.ISL_LEN_TOKENS) for _ in range(num_isls)]

    def create_prompts(self):
        prompts = [{"req" : self.get_rand_req(self.KVC_LEN_TOKENS), "keys" : []} for _ in range(self.CONVERSATIONS * self.STEPS)]
        return prompts

    async def execute_single_request_in_llm(self,  req,  indx):
        prompt = {"prompt_token_ids": req}
        async for output in self.engine.generate(prompt, sampling_params=self.sampling_params, request_id=indx):
            pass 

    async def enter_new_request(self, pp, isls, indx):
        import time
        global_start_time = time.time()
        
        for step_idx, p in enumerate(pp):
            step_start_time = time.time()
            
            if global_vars.backend:
                self.statistics.curr_disk_inflights += 1
                for i, mem_obj in enumerate(await global_vars.backend.prefetch_async(p['keys'])):
                    if mem_obj:
                        global_vars.backend.add_to_prefetched(p['keys'][i], mem_obj) 
                self.statistics.curr_disk_inflights -= 1
            self.statistics.curr_llm_inflights += 1
            
            msg = f"STARTING ASYNC EXECUTION INDX {indx} STEP {step_idx}"
            print(f"{BOLD_RED}{msg}{RESET}")
            log_to_pid_file(msg)
            
            await self.execute_single_request_in_llm(p["req"] + isls[indx-1], indx)
            
            msg = f"FINISHING ASYNC EXECUTION INDX {indx} STEP {step_idx}"
            print(f"{BOLD_RED}{msg}{RESET}")
            log_to_pid_file(msg)

            if global_vars.backend:
                for key in p['keys']:
                    global_vars.backend.remove_from_prefteched(key) 
            self.statistics.curr_llm_inflights -= 1
            
            step_time = time.time() - step_start_time
            msg = f"STEP {step_idx} (REQUEST {indx}) TOOK {step_time:.4f} SECONDS"
            print(f"{BRIGHT_YELLOW}{msg}{RESET}")
            log_to_pid_file(msg)
        
        global_time = time.time() - global_start_time
        msg = f"TOTAL CONVERSATION (REQUEST {indx}) TOOK {global_time:.4f} SECONDS"
        print(f"{BRIGHT_GREEN}{msg}{RESET}")
        log_to_pid_file(msg)


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
        
    def convert_prompts_to_conversations(self, raw_prompts):
        prompts = []
        for i in range(len(raw_prompts)):
            if i % self.STEPS == 0:
                prompts.append([raw_prompts[i]])
            else:
                last = prompts[-1][-1]  # Get the last dict from the current conversation
                curr = raw_prompts[i]
                next = {'req':last['req'] + curr['req'], 'keys':last['keys'] + curr['keys'] }
                prompts[-1].append(next)
        return prompts

    async def run_benchmark(self, MAX_INFLGITHS, NUM_ITERATIONS):
        raw_prompts = self.create_prompts()
        isls = self.create_isls(NUM_ITERATIONS)
        await self.add_hash_keys_to_prompts(raw_prompts)
        prompts = self.convert_prompts_to_conversations(raw_prompts)
        running, inflights = set(), 0
        start_time = time.time() 
        
        for i in range(1, NUM_ITERATIONS):
            running.add(asyncio.create_task(self.enter_new_request(prompts[i % len(prompts)], isls, i)))
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
    for inflights in [1]:
        print(f"{BOLD_RED}STARTING ITERATION WITH {inflights} INFLIGHTS {RESET}")
        benchmark.statistics.reset()
        benchmark.terminate = False
        benchmark.stat_task = asyncio.create_task(benchmark.update_statistics()) 
        await benchmark.run_benchmark(inflights, inflights * 5)
        benchmark.terminate = True
        await benchmark.stat_task
    


if __name__ == "__main__":
    asyncio.run(main())
