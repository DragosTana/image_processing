# Makefile for compiling CUDA program with OpenCV and OpenMP
# Compiler setup
NVCC = nvcc
CXXFLAGS =  -Xcompiler -fopenmp -Xcompiler -march=x86-64-v3 -std=c++11
LIBS = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lgomp
# Directories
INCLUDE_DIRS = -I/usr/local/include/opencv4/
LIB_DIRS = -L/usr/local/lib
# Source files
SOURCES = main.cu
# Output executable name
EXECUTABLE = convolution
# Compile rule
all: $(SOURCES)
	$(NVCC) $(SOURCES) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) $(CXXFLAGS) -o $(EXECUTABLE)
# Clean rule
clean:
	rm -f $(EXECUTABLE)
