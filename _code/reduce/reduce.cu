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

  //  for (int offset = ceil(blockDim.x / 2.f); offset > 0; offset >>= 1) {
  //if (thread_id < offset) {
  //    s_data[thread_id] += s_data[thread_id + offset];
  //  }
  //  __syncthreads();
  //}

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

  const int nr_of_elements = 100000;

  // dimensions of grid and blocks
  const dim3 blockSize(256);
  const dim3 gridSize(ceil(nr_of_elements/ ((float) blockSize.x)));

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
  long *h_in = new long[nr_of_elements];
  for (int i = 0; i < nr_of_elements; i++) {
    h_in[i] = i;
  }

  // allocate and copy input data to device
  long *d_in;
  checkCudaErrors(cudaMalloc(&d_in, sizeof(long) * nr_of_elements));
  checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(long) * nr_of_elements, cudaMemcpyHostToDevice));

  // allocate device memory for intermediate results
  long *d_out;
  checkCudaErrors(cudaMalloc(&d_out, sizeof(long) * gridSize.x));

  long *h_result = new long[1];
  long *d_result;
  checkCudaErrors(cudaMalloc(&d_result, sizeof(long)));

  // run kernel
  reduceKernel<<<gridSize, blockSize, blockSize.x * sizeof(long)>>>(d_in, d_out, nr_of_elements);
  checkCudaErrors(cudaGetLastError());

  reduceKernel<<<1, gridSize2,  gridSize2.x * sizeof(long)>>>(d_out, d_result, gridSize.x);
  checkCudaErrors(cudaGetLastError());

  // copy output from device to host
  checkCudaErrors(cudaMemcpy(h_result, d_result, sizeof(long), cudaMemcpyDeviceToHost));

  // print output array
  std::cout << "Sum: " << h_result[0] << std::endl;
}
