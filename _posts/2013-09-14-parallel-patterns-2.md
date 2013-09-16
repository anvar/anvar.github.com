---
layout: post
title: Parallel Patterns, Part 2
category: posts
---

*This post will continue to look at various patterns for creating scalable massively parallel algorithms. For earlier posts in this series, see [Introduction to CUDA]({{ site.url }}/posts/introduction-to-cuda) and [Parallel Patterns, Part 1]({{ site.url }}/posts/parallel-patterns-1).*

Stencil
-------

*Stencil* is quite similar to *gather*, but uses a fixed pattern of neighboring elements to calculate the resulting output element. For example, one common application of *stencil* is create a blurred version of an image. The input to the algorithm is a 2D array representing the image to be blurred, while the output is another 2D array representing the blurred image. There are multiple stencils that can be applied, but one commonly used is the 2D von Neumann stencil. The algorithm applies the stencil to each input pixel, reading values from every pixel covered by the stencil, and uses those values to calculate the resulting output pixel.

<img src="{{ site.url }}/assets/img/algo-von-neumann.png" width="118" height="118" class="center caption"/>
<div class="caption">2D von Neumann Stencil</div>

Given how a *stencil* works it is easy to see that every element in the input data will be read multiple times simultaneously by multiple threads as the stencil is applied to its surrounding elements. Therefore, similarly to the patterns discussed in the [previous part]({{ site.url }}/posts/parallel-patterns-1), a naive *stencil* requires a significant amount of global memory calls, and can benefit greatly from using shared memory to reduce global memory access.

<img src="{{ site.url }}/assets/img/algo-stencil.png" width="309" height="156" class="center caption"/>
<div class="caption">Applying a 2D von Neumann Stencil on an image. The darkest areas illustrate the current pixels, while the lighter areas show the pixels covered by the stencil when applied to current the pixels</div>

The code below implements a simple stencil pattern that, given a 2D array, generates a resulting 2D array by summing each element with the elements covered by the stencil.

{% highlight cuda %}

__global__ void stencilKernel(const int* const g_in,
                              int* const g_out,
                              const unsigned int rows,
                              const unsigned int cols)
{

  const int stencilWidth = 3;
  const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                       blockIdx.y * blockDim.y + threadIdx.y);

  if (thread_2D_pos.x >= cols || thread_2D_pos.y >= rows) {
    return;
  }

  int output_elem = 0;

  for (int i = 0; i < stencilWidth; i++) {
    int elem_x = min(max(thread_2D_pos.x - stencilWidth / 2 + i, 0), cols - 1);
    output_elem += g_in[elem_x + thread_2D_pos.y * cols];
  }

  for (int i = 0; i < stencilWidth; i++) {
    int elem_y = min(max(thread_2D_pos.y - stencilWidth / 2 + i, 0), rows - 1);
    output_elem += g_in[thread_2D_pos.x + elem_y * cols];
  }

  output_elem -= g_in[thread_2D_pos.x + thread_2D_pos.y * cols];
  g_out[thread_2D_pos.x + thread_2D_pos.y * cols] = output_elem;
}

{% endhighlight %}

Run the following code to execute the kernel.

{% highlight cuda %}

const dim3 blockSize(12,8);
const dim3 gridSize(ceil(cols/ (float) blockSize.x), ceil(rows/ (float) blockSize.y));

stencilKernel<<<gridSize, blockSize>>>(d_in, d_out, rows, cols);

{% endhighlight %}

As the data we are working with is arranged in a two-dimensional shape it makes sense to arrange our blocks and threads in a similar manner. That way we get a natural way of assigning threads to elements and making sure that all elements get processed. The exact dimensions used in this example are not very important in of themselves, but it is worth pointing out that they follow the rules outlined in [my first post]({{ site.url }}/posts/introduction-to-cuda) by having the block size be a multiple of 32 (*12 x 8*). The grid dimensions are just ensuring that we have enough blocks to process all input data. The fact that CUDA allows you to arrange threads in a way that is tailored to the shape of the input data greatly simplifies algorithm development.

Lets go through the code in a bit more detail to see what it does.

{% highlight cuda %}

__global__ void stencilKernel(const int* const g_in,
                              int* const g_out,
                              const unsigned int rows,
                              const unsigned int cols)

{% endhighlight %}

Because the kernel receives a continuous block of input data, but expects a 2D array, the data is stored in a row-major order, which stores the rows after each other as opposed to column-major order which stores the columns after each other. The parameters `rows` and `cols` are for keeping track of the dimensions of the 2D array.

{% highlight cuda %}

  const int stencilWidth = 3;
  const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                       blockIdx.y * blockDim.y + threadIdx.y);

  if (thread_2D_pos.x >= cols || thread_2D_pos.y >= rows) {
    return;
  }

{% endhighlight %}

The constant `stencilWidth` indicates the width and height of the 2D von Neumann stencil, so a `stencilWidth` of *3* would create a stencil just like the one earlier illustrated.

We then use the built-in function `make_int2` to create a tuple holding the threads' coordinates in the two-dimensional space, and finally make sure that neither of the threads work on data that is outside our intended boundaries.

{% highlight cuda %}

for (int i = 0; i < stencilWidth; i++) {
  int elem_x = min(max(thread_2D_pos.x - stencilWidth / 2 + i, 0), cols - 1);
  output_elem += g_in[elem_x + thread_2D_pos.y * cols];
}

{% endhighlight %}

This is where the horizontal part of the stencil is applied. The `min-max` functions ensure that we clamp our data if the stencil spills outside of the input data boundaries, i.e. if the stencil tries to read outside of the input array it will get the last correct value in that dimension. The loop will simply sum the current element, the element prior to that, and the element after.

<img src="{{ site.url }}/assets/img/algo-stencil2.png" width="215" height="261" class="center caption"/>
<div class="caption">An illustration of how cases where the stencil spills outside of the input data boundaries are handled</div>

{% highlight cuda %}

for (int i = 0; i < stencilWidth; i++) {
  int elem_y = min(max(thread_2D_pos.y - stencilWidth / 2 + i, 0), rows - 1);
  output_elem += g_in[thread_2D_pos.x + elem_y * cols];
}

{% endhighlight %}

Applying the vertical part of the stencil is treated in the same way as before with clamping on the edges.

{% highlight cuda %}

output_elem -= g_in[thread_2D_pos.x + thread_2D_pos.y * cols];
g_out[thread_2D_pos.x + thread_2D_pos.y * cols] = output_elem;

{% endhighlight %}

As the stencil is applied one dimension at the time the element in the middle will be applied twice and needs to be removed from the sum, which is then written to the threads' corresponding output location. No synchronization is needed as the writes are uncontended.
