from itertools import product
import subprocess

for (is_kv_in_hbm, tp, batch_size) in product([True, False], [1,2,4,8], [1,2,4,8]):
    subprocess.run(["python3", "test.py", str(is_kv_in_hbm), str(tp), str(batch_size)])
   

