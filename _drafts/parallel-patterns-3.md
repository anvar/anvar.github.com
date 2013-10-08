---
layout: post
title: Parallel Patterns, Part 3
category: posts
---

*This is the fourth post in my ongoing series about CUDA and parallel patterns. For earlier posts in this series, see [Introduction to CUDA]({{ site.url }}/posts/introduction-to-cuda), [Parallel Patterns, Part 1]({{ site.url }}/posts/parallel-patterns-1), and [Parallel Patterns, Part 2]({{ site.url }}/posts/parallel-patterns-2).*

Reduce
------

*Reduce* is interesting because it is the first parallel pattern we examine that requires cooperation between processors, where some of the sub-computation needs to be completed before the algorithm can proceed. At its core *reduce* takes two inputs. The first one is a set of elements to be reduced, and the second one is a reduction operator. The operator needs to be both binary, taking two inputs and producing one output, and associative where the order of the operations does not matter as long as the order of the operands is not changed. A good example of a valid operator would be *plus*, as it can take two inputs and produce one output, and it is also associative in that *(1 + 2) + 3 == 1 + (2 + 3)*.

When performing a serial *reduce*, every step of the computation is dependent on the previous one. This creates a problem when trying to parallelise the algorithm. If every computation is dependent on the result of the previous computation, then the algorithm is intrinisically not able to run any faster regardless of the number of threads we throw at it. The key lies in the associative nature of the reduction operator, which allows us to reorder the operations as long as the order of the operands is not changed. What this means in practise is that we can think of our reduction like a tree where threads can work on their individual part of the computation independently.

<img src="{{ site.url }}/assets/img/algo-reduce1.png" width="435" height="265" class="center caption"/>
<div class="caption">A tree-based approach of performing a <i>reduce</i> over a set of integers</div>

The parallel approach does come with a problem though; how do we communicate that a certain level of the computation is completed and that the next level can begin? One effective way of doing it is to decompose the computation into multiple kernel invocations. Start by launching the kernel with number of thread blocks. Each thread block will be responsible for performing a reduce over a set of values, using intra-block synchronization to ensure correctness. Once a thread block is done it writes the result to a memory location. When all thread blocks are finished the kernel is launched again but this time with only one thread block that will perform a final reduce over the results that the previous thread blocks computed.

<img src="{{ site.url }}/assets/img/algo-reduce2.png" width="533" height="342" class="center caption"/>
<div class="caption">Illustrating the kernel decomposition of a reduce operation</div>

The code below implements a simple reduce that will sum an array of values. Although the code works, the implementation is far from optimal having interleaved addressing and divergent branching in the kernel. In a future blog post I will go through the process of optimizing a reduce operation, but for now lets focus on understanding the algorithm first.

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

__global__ void reduceKernel(const long* const g_in, long* const g_out, const size_t n)
{
  extern __shared__ long s_data[];

  unsigned int thread_id = threadIdx.x;
  unsigned int global_id = (blockIdx.x * blockDim.x) + thread_id;

  long x = 0;
  if (global_id < n) {
    x = g_in[global_id];
  }
  s_data[thread_id] = x;
  __syncthreads();

  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (thread_id % (2 * s) == 0) {
      s_data[thread_id] += s_data[thread_id + s];
    }
    __syncthreads();
  }

  if (thread_id == 0) {
    g_out[blockIdx.x] = s_data[0];
  }
}

int main(int argc, char **argv) {

  const int N = 100000;

  // dimensions of grid and blocks
  const dim3 blockSize(256);
  const dim3 gridSize(ceil(N/ ((float) blockSize.x)));

  unsigned int v = gridSize.x;

  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;

  const dim3 gridSize2(v);

  // initialize input array with dummy data
  long *h_in = new long[N];
  for (int i = 0; i < N; i++) {
    h_in[i] = i;
  }

  // allocate and copy input data to device
  long *d_in;
  checkCudaErrors(cudaMalloc(&d_in, sizeof(long) * N));
  checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(long) * N, cudaMemcpyHostToDevice));

  // allocate device memory for intermediate results
  long *d_out;
  checkCudaErrors(cudaMalloc(&d_out, sizeof(long) * gridSize.x));

  // allocate device memory for final result
  long *h_result = new long[1];
  long *d_result;
  checkCudaErrors(cudaMalloc(&d_result, sizeof(long)));

  reduceKernel<<<gridSize, blockSize, blockSize.x * sizeof(long)>>>(d_in, d_out, N);
  checkCudaErrors(cudaGetLastError());

  reduceKernel<<<1, gridSize2,  gridSize2.x * sizeof(long)>>>(d_out, d_result, gridSize.x);
  checkCudaErrors(cudaGetLastError());

  // copy output from device to host
  checkCudaErrors(cudaMemcpy(h_result, d_result, sizeof(long), cudaMemcpyDeviceToHost));

  // print output array
  std::cout << "Sum: " << h_result[0] << std::endl;
}

{% endhighlight %}

The code block starts off with the usual `define` and utility function for CUDA error checking. In my [introduction to CUDA]({{ site.url }}/posts/introduction-to-cuda/), I go through the function and its purpose in more detail so I will not repeat it here.

{% highlight cuda %}
__global__ void reduceKernel(const long* const g_in, long* const g_out, const size_t n)
{
  extern __shared__ long s_data[];
{% endhighlight %}

Next up is the kernel, which begins by defining a pointer to some shared memory. Shared memory differs from global memory in that it sits directly on the multiprocessors themselves. This greatly reduces the memory access latency but also means that threads from different thread blocks are unable to access the same shared memory. I am using the unsized declaration of the shared memory here which allows me to specify the amount of shared memory when launching the kernel as opposed to during compile time, which is specified by the third kernel launch parameter, like so `reduceKernel<<<gridSize, blockSize, sharedMemorySize>>>(..);`. The shared memory will act as a temporary cache for speeding up our reduction, and allows us to avoid going to global memory every iteration.

{% highlight cuda %}
long x = 0;
if (global_id < n) {
  x = g_in[global_id];
}
s_data[thread_id] = x;
__syncthreads();
{% endhighlight %}

The first stage of the algorithm is to transfer the input data from global memory to shared memory to speed up any subsequent access. If the input data cannot fully occupy the shared memory space the extra memory slots are zeroed out to prevent them from interfering with the calculation. Before proceeding to the actual calculation we want to make sure that all input data we need has been transferred, otherwise the result might not be what we expect it to be. `__syncthreads()` creates a barrier forcing all threads in a block to reach the barrier before any of them can continue, which means that once we are past the barrier we can be sure that the data we need has been transferred.

{% highlight cuda %}
for (unsigned int s = 1; s < blockDim.x; s *= 2) {
  if (thread_id % (2 * s) == 0) {
    s_data[thread_id] += s_data[thread_id + s];
  }
  __syncthreads();
}
{% endhighlight %}


Before continuing going through the code it might be good to step back and understand exactly what it is that the algorithm is doing.

<img src="{{ site.url }}/assets/img/algo-reduce3.png" width="543" height="396" class="center caption"/>
<div class="caption">Visualising summing elements in a list using reduce.<br/><i>(The numbers in the circles represent the thread id.)</i></div>
