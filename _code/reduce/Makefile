NVCC=nvcc

CUDA_INCLUDEPATH=/Developer/NVIDIA/CUDA-5.0/include
CUDA_LIBPATH=/Developer/NVIDIA/CUDA-5.0/lib

NVCC_OPTS=-O3 -arch=sm_10 -Xcompiler -Wall -Xcompiler -Wextra -m64

GCC_OPTS=-O3 -Wall -Wextra -m64

all: reduce

reduce: reduce.o
	g++ $(GCC_OPTS) -o reduce reduce.o -L$(CUDA_LIBPATH) -lcudart

reduce.o: reduce.cu
	$(NVCC) -c reduce.cu -I$(CUDA_INCLUDEPATH) $(NVCC_OPTS)
	
clean:
	rm -f *.o reduce
	find . -type f -name '*.exr' | grep -v memorial | xargs rm -f
