import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

output = []
result = []

executable_path = "./convolution"
dimension = "1024"
kernel = "blur"
algorithm = ["seq", "omp", "cuda"]

for algo in algorithm:
    for i in range(20):
    
        args = [executable_path, dimension, kernel, algo]
        completed_process = subprocess.run(args, stdout=subprocess.PIPE, text=True)
        output.append(float(completed_process.stdout))
        
    result.append(np.mean(output))
    output = []

speedup = result[0] / result

print("Speedup: ", speedup)

