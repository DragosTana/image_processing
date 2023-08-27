import subprocess
import numpy as np
import matplotlib.pyplot as plt

output = []
result = []

executable_path = "./convolution"
dimension = "512"
kernel = "blur"
algorithm = ["seq", "omp"]

for algo in algorithm:
    for i in range(50):
    
        args = [executable_path, dimension, kernel, algo]
        completed_process = subprocess.run(args, stdout=subprocess.PIPE, text=True)
        output.append(float(completed_process.stdout))
        
    result.append(np.mean(output))
    output = []

speedup = result[0] / result

plt.bar(algorithm, speedup)
plt.show()

