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

The parallel approach does come with a problem though; how do we communicate that a certain level of the computation is completed and that the next level can begin? One effective way of doing it is to decompose the computation into multiple kernel invocations. Start by launching the kernel with a number of thread blocks. Each thread block will be responsible for performing a reduction over a set of values, using intra-block synchronization to ensure correctness. Once a thread block is done it writes the result to a memory location. When all thread blocks are finished the kernel is launched again but this time with only one thread block that will perform one final reduction over the results that the previous thread blocks computed.

<img src="{{ site.url }}/assets/img/algo-reduce2.png" width="533" height="342" class="center caption"/>
<div class="caption">Illustrating the kernel decomposition of a <i>reduce</i> operation</div>

The code below implements a simple *reduce* that will sum an array of values. Although the code works, the implementation is far from optimal with plenty of potential for improvement that can hopefully be covered in a future blog post. For now, lets focus on understanding the algorithm first.

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

  for (unsigned int distance = 1; distance < blockDim.x; distance *= 2) {
    bool is_active = (thread_id % (2 * distance) == 0);
    if (is_active) {
      s_data[thread_id] += s_data[thread_id + distance];
    }
    __syncthreads();
  }

  if (thread_id == 0) {
    g_out[blockIdx.x] = s_data[0];
  }
}

int main(int argc, char **argv) {

  const int nr_of_elements = 100000;

  // dimensions of grid and blocks
  const dim3 blockSize(256);
  const dim3 gridSize(ceil(nr_of_elements/ ((float) blockSize.x)));

  unsigned int v = gridSize.x;

  // see http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;

  const dim3 gridSize2(v);

  // initialize input array with dummy data
  long *h_in = new long[nr_of_elements];
  for (int i = 0; i < nr_of_elements; i++) {
    h_in[i] = i;
  }

  // allocate and copy input data to device
  long *d_in;
  checkCudaErrors(cudaMalloc(&d_in, sizeof(long) * nr_of_elements));
  checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(long) * nr_of_elements, cudaMemcpyHostToDevice));

  // allocate device memory for intermediate results
  long *d_intermediate;
  checkCudaErrors(cudaMalloc(&d_intermediate, sizeof(long) * gridSize.x));

  long *h_out = new long[1];
  long *d_out;
  checkCudaErrors(cudaMalloc(&d_out, sizeof(long)));

  // run kernel
  reduceKernel<<<gridSize, blockSize, blockSize.x * sizeof(long)>>>(d_in, d_intermediate, nr_of_elements);
  checkCudaErrors(cudaGetLastError());

  reduceKernel<<<1, gridSize2,  gridSize2.x * sizeof(long)>>>(d_intermediate, d_out, gridSize.x);
  checkCudaErrors(cudaGetLastError());

  // copy output from device to host
  checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(long), cudaMemcpyDeviceToHost));

  // print output array
  std::cout << "Sum: " << h_out[0] << std::endl;
}
{% endhighlight %}

The code block starts off with the usual `define` and utility function for CUDA error checking. In my [introduction to CUDA]({{ site.url }}/posts/introduction-to-cuda/), I go through the function and its purpose in more detail so I will not repeat it here.

{% highlight cuda %}
__global__ void reduceKernel(const long* const g_in, long* const g_out, const size_t n)
{
  extern __shared__ long s_data[];
{% endhighlight %}

Next up is the kernel, which begins by defining a pointer to some shared memory. Shared memory differs from global memory in that it sits directly on the multiprocessors themselves. This greatly reduces the memory access latency but also means that threads from different thread blocks are unable to access the same shared memory. I am using the unsized declaration of the shared memory here, which allows me to specify the amount of shared memory when launching the kernel, and is specified by the third kernel launch parameter, like so

{% highlight cuda %}
reduceKernel<<<gridSize, blockSize, sharedMemorySize>>>(..);
{% endhighlight %}

The alternative way of reserving shared memory would be to specify it directly in the kernel but that would also mean that the amount would have to be known at compile time. Without using shared memory as a cache our algorithm would have to go to global memory every iteration, which would have a very negative impact on the performance of the algorithm.

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
for (unsigned int distance = 1; distance < blockDim.x; distance *= 2) {
  bool is_active = (thread_id % (2 * distance) == 0);
  if (is_active) {
    s_data[thread_id] += s_data[thread_id + distance];
  }
  __syncthreads();
}

if (thread_id == 0) {
  g_out[blockIdx.x] = s_data[0];
}
{% endhighlight %}

The algorithm divides the threads into active threads and inactive threads. An active thread is responsible for adding the value of the next subsequent inactive thread, defined by a *distance*, to their own. The *distance* starts at *1* and is doubled every iteration, which means that the number of threads need to be a power of two so that every time the distance is doubled it provides the same number of active and inactive threads. Finally the calculated sum is written to the output memory location indicated by the block id. Although the algorithm might sound complicated it is in fact quite straightforward. The figure below might provide a better illustration of how the algorithm works.

<img src="{{ site.url }}/assets/img/algo-reduce3.png" width="543" height="396" class="center caption"/>
<div class="caption">Visualising summing elements in a list using <i>reduce</i>.<br/><i>(The numbers in the circles represent the thread id)</i></div>

Although the algorithm produces the right result, as you can imagine, the implementation is far from optimal with problems such poor data locality and branch divergence. I hope to be able to go through the process of optimizing this kernel in a future blog post, but for now lets take a look at some things we have to do before launching the algorithm.

{% highlight cuda %}
const dim3 blockSize(256);
const dim3 gridSize(ceil(nr_of_elements/ ((float) blockSize.x)));

unsigned int v = gridSize.x;

// see http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
v--;
v |= v >> 1;
v |= v >> 2;
v |= v >> 4;
v |= v >> 8;
v |= v >> 16;
v++;

const dim3 gridSize2(v);
{% endhighlight %}

The first two lines should be quite familar by now, but the lines following them might look a bit strange. Earlier, when going through the kernel algorithm, I mentioned that it was important that the number of threads used needed to be a power of two. We ensure that this is the case for the first kernel launch by using a fixed amount of threads and varying the number of blocks instead. The second kernel launch, however, cannot have a fixed number of threads as it needs at least one thread for each block, and because the number of blocks is dependent on the number of values we cannot know at compile time how many threads we need. In order to get the right amount of threads we first need to take the number of blocks and then round that number up to the nearest number that is a power of two, i.e. *205* becomes *256*. I am not going to go in to too much detail about how the rounding process works, but for more information and other interesting bit twiddling hacks check out [bithacks](http://graphics.stanford.edu/~seander/bithacks.html‎). With the rounding in place we can be certain that we will have a correct number of threads for launching the kernel the second time.

{% highlight cuda %}
// initialize input array with data ranging from [0,nr_of_elements)
long *h_in = new long[nr_of_elements];
for (int i = 0; i < nr_of_elements; i++) {
  h_in[i] = i;
}

// allocate and copy input data to device
long *d_in;
checkCudaErrors(cudaMalloc(&d_in, sizeof(long) * nr_of_elements));
checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(long) * nr_of_elements, cudaMemcpyHostToDevice));

// allocate device memory for intermediate results
long *d_intermediate;
checkCudaErrors(cudaMalloc(&d_intermediate, sizeof(long) * gridSize.x));

long *h_out = new long[1];
long *d_out;
checkCudaErrors(cudaMalloc(&d_out, sizeof(long)));
{% endhighlight %}

Here we initialize our data set with a monotonically increasing number series starting at *0*, and we also reserve the necessary space in the global memory of the GPU. The memory will be used for holding our input data, our final results, and any intermediate data between kernel invocations.

{% highlight cuda %}
reduceKernel<<<gridSize, blockSize, blockSize.x * sizeof(long)>>>(d_in, d_intermediate, nr_of_elements);
checkCudaErrors(cudaGetLastError());

reduceKernel<<<1, gridSize2,  gridSize2.x * sizeof(long)>>>(d_intermediate, d_out, gridSize.x);
checkCudaErrors(cudaGetLastError());

// copy output from device to host
checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(long), cudaMemcpyDeviceToHost));

std::cout << "Sum: " << h_out[0] << std::endl;
{% endhighlight %}

Finally we are ready to launch the kernels, with the results of the first invocation used as input data to the second one. After both kernels have executed the total sum is then transfered back to the host and printed to `stdout`, which concludes our reduction example.
