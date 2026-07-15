import aie.iron as iron
import numpy as np
import time
import os
import psutil

process = psutil.Process(os.getpid())
print(f"Initial Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")

t0 = time.perf_counter()
tensors = []
for i in range(280):
    data = np.zeros((4096, 4096), dtype=np.uint8)
    t = iron.tensor(data.reshape(-1), dtype=np.uint8, device="npu") # approx 16MB each. Total = 4.4GB
    tensors.append(t)
t1 = time.perf_counter()

print(f"Allocated 280 iron.tensors in {t1 - t0:.2f}s")
print(f"Final Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
