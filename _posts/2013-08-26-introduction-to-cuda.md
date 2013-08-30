---
layout: post
title: "Introduction to CUDA"
category: posts
---

GPUs; once strongly associated with gaming are now becoming common place in industry where frameworks like CUDA and OpenCL are exposing their power for general applications with serious needs for computation. The key to their power is their parallelism. Where a standard CPU has between 4-8 cores, a GPU like the Nvidia GTX Titan readily comes with 2688 cores. The key, therefore, to successfully unlock their power is two-fold. The first part is to have an algorithm that can scale to the required level of parallelism. The second part, which may be less obvious, is to maximize GPU occupancy by carefully partitioning the computation to work with the hardware layout. In order to do that we must first look at what a GPU looks like at logical hardware level.

The newest NVIDIA compute architecture is Kepler (compute capability 3.x), but here I will focus on describing a Fermi-based compute architecture (compute capability 2.x) as it is the one I have most familiarity with, and the one that Amazon EC2 uses in their Cluster GPU instance types.

<img src="{{ site.url }}/assets/img/hw-layout.png" width="568" height="338" alt="Fermi-based GPU Hardware Layout" class="center caption"/>
<div class="caption">Fermi-based GPU Hardware Layout</div>

A Fermi-based GPU is based upon 16 Streaming Multiprocessors (SM) positioned around a common L2 cache, with each SM containing 32 cores, registers, and a small chunk of shared memory. Surrounding the SMs are six 64-bit memory partitions, for a 384-bit memory interface, supporting up to a total of 6GB of GDDR5 DRAM memory. The GigaScheduler provides a GPU-level scheduler distributing thread blocks to SMs internal schedulers, while the Host Interface connects through PCI-Express to the host system.

<img src="{{ site.url }}/assets/img/sm-layout.png" width="232" height="552" alt="Fermi Streaming Multiprocessor" class="center caption"/>
<div class="caption">Fermi Streaming Multiprocessor</div>

Because an SM contains 32 cores it can, as such, only execute a maximum of 32 threads at any given time, which in CUDA-speak is called a warp. Every thread in a warp is executed in SIMD lockstep fashion, executing the same instruction but using its private registers to perform the requested operation. To get the most performance out of a GPU it is therefore important to understand warps, and how to write warp-friendly code. Warps are the primary way a GPU can hide latency, so if an operation will take a long time to perform - such as fetching from global memory - the warp scheduler will park the warp and schedule a different one. Once the memory access returns the warp will be rescheduled. By optimizing the memory access within warps the memory access can be coalesced with one call fetching data for several threads within the warp, and thus greatly reduce the overall cost of memory latency. Additionally, as the instructions are executed in lockstep, it is important to try to avoid conditional code which forces the threads within a warp to execute different branches as that can also have a significant impact on the time it takes to complete a warp.

Programming model
-----------------
A program designed to run on a GPU is called a kernel, and in CUDA the level of parallelism for a kernel is defined by the grid size and the block size. The grid size defines the number of blocks and the shape of the cube that the blocks are distributed within. Ever since compute capability 2.0 the grid size can be specified in three dimensions, whilst in earlier versions being restricted to only two. The block size follows a similar model but has always had the ability to be specified in three dimensions.

<img src="{{ site.url }}/assets/img/cuda-grid.png" width="376" height="283" class="center caption"/>
<div class="caption">Possible arrangement of CUDA blocks and threads</div>

How to choose size
------------------
In general, outside of the maximum allowed dimensions for blocks and grids, you want to size your blocks/grid to match your data and simultaneously maximize occupancy, that is, how many threads are active at one time. The major factors influencing occupancy are shared memory usage, register usage, and thread block size. Another factor to consider is that threads get scheduled in warps, so a block size should always be a multiple of 32, otherwise at least one warp will be scheduled that is not making use of all the cores in the SM. Picking the right dimensions is somewhat of a black art, as it can be GPU/kernel/data shape/algorithm dependent. Sometimes, for example, it makes sense to perform a bit more work in an individual thread to minimize the number of blocks that need to be scheduled. Therefore it is always good to experiment with various sizes to see what the impact is on your specific kernel.

When a thread is executing a kernel there are a few variables that CUDA exposes that can help with identifying which thread it is:

* `gridDim.{x,y,z}` - *The dimensions of the grid*
* `blockDim.{x,y,z}` - *The dimensions of the block*
* `blockIdx.{x,y,z}` - *The index of the current block within the grid*
* `threadIdx.{x,y,z}` - *The index of the current thread with the block*

CUDA Hello World
----------------

This is a simple example of how to write a CUDA program. It will take an input vector of integers and, with the help of a GPU, return an output vector consisting of the elements in the input vector times themselves. So the following input vector `[1, 2, 3, 4]` becomes `[1, 4, 9, 16]`. Below is the program in its entirety, and the following sections will drill down into more detail.

To run the program make sure that the CUDA SDK is installed and that you can access the `nvcc` executable therein. Then execute the following lines in a terminal to compile and execute the program, given that the code is in a file named `hello.cu`.

{% highlight bash %}
nvcc hello.cu -o hello.out
./hello.out
{% endhighlight %}

{% highlight cuda %}
#include <iostream>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

__global__ void helloKernel(const int* const g_in, int* const g_out, const unsigned int n)
{
  unsigned int global_id = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (global_id < n) {
    g_out[global_id] = g_in[global_id] * g_in[global_id];
  }
}

int main(int argc, char **argv) {

  const int nr_of_elements = 10;

  // initialize input array with dummy data
  int *h_in = new int[nr_of_elements];
  for (int i = 0; i < nr_of_elements; i++) {
    h_in[i] = i;
  }

  // allocate and copy input data to device
  int *d_in;
  checkCudaErrors(cudaMalloc(&d_in, 2 * sizeof(int) * nr_of_elements));
  checkCudaErrors(cudaMemcpy(d_in,
                             h_in,
                             sizeof(int) * nr_of_elements,
                             cudaMemcpyHostToDevice));

  // allocate array for output
  int *h_out = new int[nr_of_elements];
  int *d_out;
  checkCudaErrors(cudaMalloc(&d_out, sizeof(int) * nr_of_elements));

  // dimensions of grid and blocks
  const dim3 blockSize(128);
  const dim3 gridSize(256);

  // run kernel
  helloKernel<<<gridSize, blockSize>>>(d_in, d_out, nr_of_elements);

  // copy output from device to host
  checkCudaErrors(cudaMemcpy(h_out,
                             d_out,
                             sizeof(int) * nr_of_elements,
                             cudaMemcpyDeviceToHost));

  // print output array
  for(int i = 0; i < nr_of_elements; i++) {
    std::cout << h_out[i] << ",";
  }
  std::cout << std::endl;
}
{% endhighlight %}

So lets drill down into the different parts that comprise this program to learn more about what they do.

{% highlight cuda %}
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
{% endhighlight %}

First off is a utility function that is used to wrap any CUDA call to make sure that the call was successful. This is very important as we risk introducing garbage data into our calculation if continuing execution without verifying the result. There are a few ways to do this, this particular implementation was copied from the excellent Udacity course [CS344](https://www.udacity.com/course/cs344), for a more detailed example look at `helper_cuda.h` in `${CUDA_SDK}/samples/common/inc`.

{% highlight cuda %}
__global__ void helloKernel(const int* const g_in, int* const g_out, const unsigned int n)
{
  unsigned int global_id = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (global_id < n) {
    g_out[global_id] = g_in[global_id] * g_in[global_id];
  }
}
{% endhighlight %}

Next up is the kernel. The kernel is the function that will be executed on the GPU and can be recognized by the `__global__` keyword, which means that it is callable from the host. A function can also be annotated with the `__device__` keyword, which will make the function callable from either another function annotated with the `__device__` keyword or a function annotated with the `__global__` keyword but not from a function executing on the host. Our kernel in this case is very simple. In the arguments list we have two pointers to the global memory on the GPU. The first one, `g_in`, is for storing the input vector and the second, `g_out`, is for storing the output vector. The `g_` prefix is just a convention to indicate that the variables refer to the global memory as opposed to shared memory or local registers. Inside the kernel we calculate our global thread id, since the `threadIdx.x` is the thread index within a block we also need to account for the block that the thread belongs to. Another point worth mentioning is that we can have more threads than there are elements in the vector so we need to make sure that we do not overstep our memory boundaries. Finally we read from the global memory address designated by `g_in` and put the computed value in `g_out`.

The `main` function is responsible for allocating memory on the GPU, transferring data forth and back, and do some rudimentary printing.

{% highlight cuda %}

  const int nr_of_elements = 10;

  // initialize input array with dummy data
  int *h_in = new int[nr_of_elements];
  for (int i = 0; i < nr_of_elements; i++) {
    h_in[i] = i;
  }

{% endhighlight %}

The function starts off with creating an array and filling it with a sequence of integers.

{% highlight cuda %}

  // allocate and copy input data to device
  int *d_in;
  checkCudaErrors(cudaMalloc(&d_in, 2 * sizeof(int) * nr_of_elements));
  checkCudaErrors(cudaMemcpy(d_in,
                             h_in,
                             sizeof(int) * nr_of_elements,
                             cudaMemcpyHostToDevice));

{% endhighlight %}

Next we allocate memory on the device using `cudaMalloc` and as mentioned previously the call is wrapped with a call to `checkCudaErrors`. Similar to `malloc` the function takes a pointer, which will be populated with the address where the allocated memory resides, and the amount of memory we wish to allocate. We also copy the data generated in previous step to the device using `cudaMemcpy` with `cudaMemcpyHostToDevice` specifying the direction of the copy.

{% highlight cuda %}

  // allocate array for output
  int *h_out = new int[nr_of_elements];
  int *d_out;
  checkCudaErrors(cudaMalloc(&d_out, sizeof(int) * nr_of_elements));

{% endhighlight %}

We also allocate memory on the GPU for the vector that will hold the results of our computations, and as expected there is no copy operation yet as we have not computed anything.

{% highlight cuda %}
  // dimensions of grid and blocks
  const dim3 blockSize(128);
  const dim3 gridSize(256);

{% endhighlight %}

The block and grid size is defined using the struct `dim3`, which holds values for the x, y, and z dimensions. Any unspecified dimension is assumed to be *1*. In this particular case there will be 256 blocks, each containing 128 threads, and both blocks and grids will be arranged in a one-dimensional fashion along the x-axis. For this simple example the exact numbers are not of great importance, but as mentioned earlier, for a more complex kernel picking the right sizes can have a great impact on performance.

{% highlight cuda %}

  // run kernel
  helloKernel<<<gridSize, blockSize>>>(d_in, d_out, nr_of_elements);

{% endhighlight %}

We are finally ready to execute our kernel, and we pass along our grid and block sizes in the `<<<>>>` part. We also pass along the pointers to the memory we have allocated.

{% highlight cuda %}

  // copy output from device to host
  checkCudaErrors(cudaMemcpy(h_out,
                             d_out,
                             sizeof(int) * nr_of_elements,
                             cudaMemcpyDeviceToHost));

  // print output array
  for(int i = 0; i < nr_of_elements; i++) {
    std::cout << h_out[i] << ",";
  }
  std::cout << std::endl;
}
{% endhighlight %}

After the kernel has finished we copy the results over from the GPU to our host by specifying the copy direction to be `cudaMemcpyDeviceToHost`, and print it to `stdout` before exiting.

Final remarks
-------------
This is the first part of an ongoing series about GPU programming using CUDA, which will be mirroring my own progress as I learn more about the subject. It is primarily intended to be a place for me to organize the knowledge I have gathered but I do hope that it can be of some value for someone else. In the next part I will start looking at the parallel algorithms that form the basis for doing useful work on a GPU.
