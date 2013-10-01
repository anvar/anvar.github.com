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



* Performance caviets
* Code
