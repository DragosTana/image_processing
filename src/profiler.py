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
    su_3 = []
    su_5 = []
    su_7 = []
    dimension = ["256", "512", "1024", "2048"]
    
    dim_time = {
        "3x3": [65.33314325093832, 81.8725552868803, 69.1972952976027, 72.79898620769971],
        "5x5": [182.37321726107746, 175.3693649554578, 166.33954778045333, 193.24668023131295],
        "7x7": [267.687157716421, 287.8663259223298, 295.01091255540456, 353.2459610586225]
    }

    
    
    x = np.arange(len(dimension))
    width = 0.25
    multiplier = 0
    
    fig, ax = plt.subplots(layout = "constrained")
    
    for attribute, value in dim_time.items():
        offest = width * multiplier
        rects = ax.bar(x + offest, value, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1
    ax.set_ylabel('SpeedUp', fontsize=20)
    ax.set_title('SpeedUp vs Dimension', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(dimension)
    ax.legend(fontsize='larger')
    fig.tight_layout()
    plt.show()
    
    
def speedup():
    output = []
    result = []

    executable_path = "./convolution"
    dimension = ["256", "512", "1024", "2048"]
    kernel = "blur"
    algorithm = ["seq", "cuda"]
    
    result = np.zeros((len(algorithm), len(dimension)))
    speedup = np.zeros(len(dimension))
    for i in range(len(dimension)):
        for j in range(len(algorithm)):
            print("Dimension: ", dimension[i], " Algorithm: ", algorithm[j])
            for n in range(10):
                args = [executable_path, dimension[i], kernel, algorithm[j]]
                completed_process = subprocess.run(args, stdout=subprocess.PIPE, text=True)
                output.append(float(completed_process.stdout))
            result[j][i] = np.mean(output)  
            output = []
            
            
            
    print(result)
    
    for i in range(len(dimension)):
        speedup[i] = result[0][i] / result[1][i]
            
    print(list(speedup))
    
if __name__ == "__main__":
    speedup()
    
    
    
    