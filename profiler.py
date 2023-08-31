import subprocess
import numpy as np
import matplotlib.pyplot as plt


def vect():
    output = []
    result = []

    executable_path = "./convolution"
    dimension = "2048"
    kernel = "blur"
    algorithm = ["seq", "omp", "cuda", "opencv"]

    for algo in algorithm:
        for i in range(50):
            args = [executable_path, dimension, kernel, algo]
            completed_process = subprocess.run(args, stdout=subprocess.PIPE, text=True)
            output.append(float(completed_process.stdout))
        result.append(np.mean(output))
        output = []

    speedup = result[0] / result

    speedup= list(speedup)
    algorithm = list(algorithm)
    
    speedup.pop(0)
    algorithm.pop(0)
    
    print("Speedup: ", speedup)

    plt.bar(algorithm, speedup)
    plt.grid()
    plt.ylabel('Speedup', fontsize=20)
    plt.xlabel('Algorithm', fontsize=20)
    plt.title('Speedup vs Algorithm', fontsize=20)
    plt.show()
    
    
def vecVsNovec():
    vec_time = [0.0033989078, 0.0067449711, 0.015695507, 0.039085473]
    novec_time =[0.0024696618, 0.008077753500000001, 0.014563581999999999, 0.041300124]
    dimension = ["256", "512", "1024", "2048"]
    
    dim_time = {
        "vec": vec_time,
        "novec": novec_time
    }

    x = np.arange(len(dimension))
    width = 0.35
    multiplier = 0
    
    fig, ax = plt.subplots(layout = "constrained")
    
    for attribute, value in dim_time.items():
        offest = width * multiplier
        rects = ax.bar(x + offest, value, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1
    ax.set_ylabel('Time (s)', fontsize=20)
    ax.set_title('Time vs Dimension', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(dimension)
    ax.legend(fontsize='larger')
    fig.tight_layout()
    plt.show()
    
    
def speedup():
    output = []
    result = []

    executable_path = "./convolution"
    dimension = "1024"
    kernel = "blur"
    algorithm = ["seq", "omp"]
    
    
    for algo in algorithm:
        print("Running: ", algo)
        for i in range(50):
            args = [executable_path, dimension, kernel, algo]
            completed_process = subprocess.run(args, stdout=subprocess.PIPE, text=True)
            output.append(float(completed_process.stdout))

        result.append(np.mean(output))
        output = []
        
    speedup = result[0] / result
    
    print("Speedup: ", speedup)
    #plot speedup and its values
    plt.bar(algorithm, speedup)
    plt.ylabel('Speedup', fontsize=20)
    plt.xlabel('Algorithm', fontsize=20)
    plt.title('Speedup vs Algorithm', fontsize=20)
    plt.show()
    
    
if __name__ == "__main__":
    vect()
    
    
    
    