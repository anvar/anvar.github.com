---
layout: post
title: "Introduction to CUDA"
category: posts
---

GPUs; once strongly associated with gaming are now becoming common place in industry where frameworks like CUDA and OpenCL are exposing their power for general applications with serious need for computation. The key to their power is their parallelism. Where a standard CPU has between 4-8 cores, a GPU like the Nvidia GTX Titan readily comes with 2688 cores. The key, therefore, to successfully unlock their power is two-fold. The first part is to have an algorithm that can scale to the required level of parallelism. The second part, which may be less obvious, is to maximize GPU occupancy by carefully partitioning the computation to work with the hardware layout. In order to do that we must first look at what a GPU looks like at logical hardware level.

The newest NVIDIA compute architecture is Kepler (compute capability 3.x), but here I will focus on a Fermi-based compute architecture (compute capability 2.x) as it is the one I have most familiarity with, and the one that Amazon EC2 uses in their Cluster GPU instance types.

A Fermi-based GPU is based upon 16 Streaming Multiprocessors (SM) positioned around a common L2 cache, with each SM containing 32 cores, registers, and a small chunk of shared memory. Surrounding the SMs are six 64-bit memory partitions, for a 384-bit memory interface, supporting up to a total of 6GB of GDDR5 DRAM memory. The GigaScheduler provides a GPU-level scheduler distributing thread blocks to SMs internal schedulers, while the Host Interface connects through PCI-Express to the host system.

Because an SM contains 32 cores it can, as such, only execute a maximum of 32 threads at any given time, which in CUDA-speak is called a warp. Every thread in a warp is executed in SIMD lockstep fashion, executing the same instruction but using its private registers to perform the requested operation. To get the most performance out of a GPU it is therefore important to understand warps, and how to write warp-friendly code. Warps are the primary way a GPU can hide latency, so if an operation will take a long time to perform - such as fetching from global memory - the warp scheduler will park the warp and schedule a different one. Once the memory access returns the warp will be rescheduled. By optimizing the memory access within warps the memory access can be coalesced with one call fetching data for several threads within the warp, and thus greatly reduce the overall cost of memory latency. Additionally, as the instructions are executed in lockstep, it is important to try to avoid conditional code which forces the threads within a warp to execute different branches as that can also have a significant impact on the time it takes to complete a warp.

Programming model
-----------------
A program designed to run on a GPU is called a kernel, and in CUDA the level of parallelism for a kernel is defined by the grid size and the block size. The grid size defines the number of blocks and the shape of the cube that the blocks are distributed within. Ever since compute capability 2.0 the grid size can be specified in three dimensions, whilst in earlier versions being restricted to only two. The block size follows a similar model but has always had the ability to be specified in three dimensions.

<img src="{{ site.url }}/assets/img/cuda-grid.png" width="305" height="290" class="center caption"/>
<div class="caption">Hello, Caption!</div>

How to choose size
------------------
In general, outside of the maximum allowed dimensions for blocks and grids, you want to size your blocks/grid to match your data and simultaneously maximise occupancy, that is, how many threads are active at one time. The major factors influencing occupancy are shared memory usage, register usage, and thread block size. Another factor to consider is that threads get scheduled in warps, so a block size should always be a multiple of 32, otherwise at least one warp will be scheduled that is not making use of all the cores in the SM. Picking the right dimensions is somewhat of a black art, as it can be GPU/kernel/data shape/algorithm dependent. Sometimes, for example, it makes sense to perform a bit more work in an individual thread to minimize the number of blocks that need to be scheduled. Therefore it is always good to experiment with various sizes to see what the impact is on your specific kernel.

When a thread is executing a kernel there are a few variables that CUDA exposes that can help with identifying which thread it is:

* gridDim.{x,y,z} - The dimensions of the grid
* blockDim.{x,y,z} - The dimensions of the block
* blockIdx.{x,y,z} - The index of the current block within the grid
* threadIdx.{x,y,z} - The index of the current thread with the block

CUDA Hello World
----------------

{% highlight cuda %}
__global__ 
void helloKernel(const float* const g_in, int* const g_out, const unsigned int n)
{
  // put this in a comment in the code instead
  unsigned int global_id = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (global_id < n) {
    g_out[global_id] = g_in[global_id] * g_in[global_id];
  }
}

int main(int argc, char **argv) {
  const dim3 blockSize(128);
  const dim3 gridSize(256);

  const int number_of_elements = 10;
  int *h_in = new float[number_of_elements];

  int *d_in;
  checkCudaErrors(cudaMalloc(&d_in, 2 * sizeof(int) * number_of_elements));
  checkCudaErrors(cudaMemcpy(d_in,
                             h_in,
                             sizeof(int) * number_of_elements, 
                             cudaMemcpyHostToDevice));

  int *h_out = new float[number_of_elements];
  int *d_out;
  checkCudaErrors(cudaMalloc(&d_out, sizeof(int) * number_of_elements));

  helloKernel<<<gridSize, blockSize>>>(d_in, d_out, number_of_elements);

  checkCudaErrors(cudaMemcpy(h_out,
                             d_out,
                             sizeof(int) * number_of_elements, 
                             cudaMemcpyDeviceToHost));
}
{% endhighlight %}





