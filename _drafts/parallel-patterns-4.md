---
layout: post
title: Parallel Patterns, Part 4
category: posts
---

*This is the second post in my ongoing series about programming a GPU using CUDA. In [my first post]({{ site.url}}/posts/introduction-to-cuda) I looked at the hardware architecture of a GPU, and how it influences the way CUDA programs are developed.*

Scan
----

* [1 2 3 4 5] -> [0 1 3 6 10]
* used for compactation, allocation, quicksort, data compression, sparse matrix computation
* takes input array and binary associative operator (and commuicative x + y = y + x)
* needs identity value (for + -> 0: x + 0 = x)
* inclusive scan = the output at each location is the elements that came before, not including the current element (first location is the identity element)
* exclusive scan = the output at each location is the elements that came bofore including the current element (first location is the first element)

Serial impl
----------------
1) Start with accumulator
2) Loop through elements
3) Take each element, apply operator, write to accumulator

Parallel
----------------
Can think of it as a reduction of some of the elements, i.e. to get the first element you have to reduce the first element, to get the second element you have to reduce over the first two elements etc (inclusive scan)

To compute all *n* outputs, simply run *n* separate reductions, which themselves can be run in parallel.

A naive scan takes O(n^2) where the serial one takes O(n)

Two options:
1) Hillis Steele (more step efficient)
More # processors than input

2) Blelloch Scan (more work efficient)
Input much larger than # processors can handle

Sequential algorithm: O(n)
Hillis Steele: O(n log(n))
Blelloch: O(n)

Hillis Steele scan
------------------

Starting with step 0, add yourself to you 2^i left neighbour

Blelloch scan
----------------

Balanced tree

2 stages; reduce, and downsweep
