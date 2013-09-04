---
layout: post
title: Parallel Algorithms, Part 1
category: posts
---

*This is the second post in my ongoing series about programming a GPU using CUDA. In [my first post]({{ site.url}}/posts/introduction-to-cuda) I looked at the hardware architecture of a GPU, and how it influences the way CUDA programs are developed.*

A GPU has very different architecture than a CPU, and that with good reason. Originally developed for speeding up computer graphic, GPU manufacturers quickly abandoned raw cycle speeds and instead focused increasing power by adding processing cores. The implications of that choice, and the massive parallelism that comes with it, is that an optimized sequential algorithm might yield very poor results in the performance department when deployed to a GPU, and the only way around it is to utilize algorithms that were designed from the ground up for parallelism.

Map
---

First up is **map**, which is arguably one of the easier to understand conceptually. Orginating from functional programming languages, **map** takes an input list *i* and a function *f*, and creates a new list *o* by applying *f* to every element in *i*. Implementing **map** in CUDA is quite straightforward. Two memory locations will represent the input list and the output list, and each thread will read from the input memory location specified by its `thread_id`, apply *f*, and write the result to the corresponding output memory location. Because each thread has a specific input and output slot, there will no contention for memory access, and thus no need for any synchronization.

<img src="{{ site.url }}/assets/img/algo-map.png" width="308" height="233" class="center caption"/>
<div class="caption">A simple <strong>map</strong> that takes an input list, applies <code>f(x) = x + 2</code>, and writes the results to an output list</div>

Below is a kernel that implements the algorithm in the figure above.

{% highlight cuda %}
__global__ void mapKernel(const int* const g_in, int* const g_out, const unsigned int n)
{
  unsigned int global_id = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (global_id < n) {
    g_out[global_id] = g_in[global_id] + 2;
  }
}
{% endhighlight %}

The code is fairly straightforward, we read from `g_in`, apply the function `f(x) = x + 2`, and write the result to `g_out`. I am also omitting any host code for transferring data in and out of the GPU. For an example of how to do that check out my [previous post]({{ site.url }}/posts/introduction-to-cuda).

Scatter
-------

**Scatter** can be seen as a variation on **map**, but instead of maintaining the one to one correspondence between input position and output position, **scatter** reads from one position and writes to one or many positions anywhere in the memory space. A simple implementation, like the one below, uses global memory for reading and writing, which makes it relatively slow due to the latency introduced by going to global memory. A more optimized version would use locality and shared memory to reduce the need for going to global memory and thus increase the performance.

<img src="{{ site.url }}/assets/img/algo-scatter.png" width="307" height="233" class="center caption"/>
<div class="caption">An example of a <strong>scatter</strong> that takes and input list and applies <code>f(x) += x</code>, and writes the result to two output locations</div>

The following kernel implements the scenario described above.

{% highlight cuda %}

__global__ void scatterKernel(const int* const g_in, int* const g_out, const unsigned int n)
{
  unsigned int global_id = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (global_id < n} {
    atomicAdd(&g_out[global_id], g_in[global_id]);
    atomicAdd(&g_out[global_id + 1], g_in[global_id]);
  }
}

{% endhighlight %}

Unfortunately there is a bit of complexity in the example above. Because we have two threads writing to the same location we need to properly synchronize when writing the result. The `atomicAdd` part will ensure that our updates processed in an atomic fashion and ensure the correctness of the operation. The rest of the code is hopefully quite straightforward.

Gather
------

as

<img src="{{ site.url }}/assets/img/algo-gather.png" width="307" height="234" class="center caption"/>
<div class="caption">Stuff</div>

ad
