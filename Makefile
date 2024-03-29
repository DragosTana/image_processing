NVCC = nvcc
CXXFLAGS = -Xcompiler -fopenmp -Xcompiler -march=native -std=c++14
LIBS = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lgomp -lopencv_imgproc
INCLUDE_DIRS = -I/usr/local/include/opencv4/
LIB_DIRS = -L/usr/local/lib
SOURCES = main.cu
EXECUTABLE = convolution
all: $(SOURCES)
	$(NVCC) $(SOURCES) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) $(CXXFLAGS) -o $(EXECUTABLE)
clean:
	rm -f $(EXECUTABLE)
