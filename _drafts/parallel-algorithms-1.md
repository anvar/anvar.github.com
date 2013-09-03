---
layout: post
title: Parallel Algorithms, Part 1
category: posts
---

*This is the second post in my ongoing series about programming a GPU using CUDA. In [my first post]({{ site.url}}/posts/introduction-to-cuda) I looked at the hardware architecture of a GPU, and how it influences the way CUDA programs are developed.*

A GPU has very different architecture than a CPU, and that with good reason. Originally developed for speeding up computer graphic, GPU manufacturers quickly abandoned raw cycle speeds and instead focused increasing power by adding processing cores. The implications of that choice, and the massive parallelism that comes with it, is that an optimized sequential algorithm might yield very poor results in the performance department when deployed to a GPU, and the only way around it is to utilize algorithms that were designed from the ground up for parallelism.

Map
---

First up is **map**, which is arguably one of the easier to understand conceptually. Orginating from functional programming languages, **map** takes an input list *i* and a function *f*, and creates a new list *o* by applying *f* to every element in *i*. Implementing **map** in CUDA is quite straightforward. Two memory locations will represent the input list and the output list, and a kernel will read from the input memory location, apply *f*, and write the result to the output memory location.

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

asd

Gather
------

ad
