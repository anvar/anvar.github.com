NVCC=nvcc

CUDA_INCLUDEPATH=/Developer/NVIDIA/CUDA-5.0/include
CUDA_LIBPATH=/Developer/NVIDIA/CUDA-5.0/lib

NVCC_OPTS=-O3 -arch=sm_10 -Xcompiler -Wall -Xcompiler -Wextra -m64

GCC_OPTS=-O3 -Wall -Wextra -m64

hello-world: hello-world.o
	g++ $(GCC_OPTS) -o hello-world hello-world.o -L$(CUDA_LIBPATH) -lcudart

hello-world.o: hello-world.cu
	$(NVCC) -c hello-world.cu -I$(CUDA_INCLUDEPATH) $(NVCC_OPTS)

clean:
	rm -f *.o hello-world
	find . -type f -name '*.exr' | grep -v memorial | xargs rm -f
