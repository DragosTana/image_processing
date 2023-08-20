# Makefile for compiling CUDA program with OpenCV and OpenMP
# Compiler setup
NVCC = nvcc
CXXFLAGS = -O2 -Xcompiler -fopenmp
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
