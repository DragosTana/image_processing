NVCC = nvcc
CXXFLAGS =  -Xcompiler -fopenmp -Xcompiler -march=native -std=c++11
LIBS = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lgomp
INCLUDE_DIRS = -I/usr/local/include/opencv4/
LIB_DIRS = -L/usr/local/lib
SOURCES = main.cu
EXECUTABLE = convolution
all: $(SOURCES)
	$(NVCC) $(SOURCES) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) $(CXXFLAGS) -o $(EXECUTABLE)
clean:
	rm -f $(EXECUTABLE)
