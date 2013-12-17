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

__global__ void scanHillisSteeleKernel(const int* const g_in, int* const g_out, const size_t n)
{
  extern __shared__ int s_data[];

  unsigned int thread_id = threadIdx.x;
  //unsigned int global_id = (blockIdx.x * blockDim.x) + thread_id;

  int  pout = 0, pin = 1;
  s_data[pout * n + thread_id] = (thread_id > 0) ? g_in[thread_id - 1] : 0;
  __syncthreads();

  for (int offset = 1; offset < n; offset *= 2)
  {
    pout = 1 - pout; // swap double buffer indices
    pin = 1 - pout;
    if (thread_id >= offset) {
      s_data[pout * n + thread_id] += s_data[pin * n + thread_id - offset];
    } else {
      s_data[pout * n + thread_id] = s_data[pin * n + thread_id];
    }
    __syncthreads();
  }
  g_out[thread_id] = s_data[pout * n + thread_id];
}

int main(int argc, char **argv) {

  const int nr_of_elements = 128;

  // dimensions of grid and blocks
  const dim3 blockSize(nr_of_elements);
  //  const dim3 gridSize(ceil(nr_of_elements/ ((float) blockSize.x)));
  const dim3 gridSize(1);

  // initialize input array with dummy data
  int *h_in = new int[nr_of_elements];
  for (int i = 0; i < nr_of_elements; i++) {
    h_in[i] = i;
  }

  // allocate and copy input data to device
  int *d_in;
  checkCudaErrors(cudaMalloc(&d_in, sizeof(int) * nr_of_elements));
  checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(int) * nr_of_elements, cudaMemcpyHostToDevice));

  int *h_out = new int[nr_of_elements];
  int *d_out;
  checkCudaErrors(cudaMalloc(&d_out, sizeof(int) * nr_of_elements));

  // run kernel
  scanHillisSteeleKernel<<<gridSize, blockSize, blockSize.x * sizeof(int)>>>(d_in, d_out, nr_of_elements);
  checkCudaErrors(cudaGetLastError());

  // copy output from device to host
  checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(int) * nr_of_elements, cudaMemcpyDeviceToHost));

  // print output array
  for(int i = 0; i < nr_of_elements; i++) {
    std::cout << h_out[i] << std::endl;
  }
}
