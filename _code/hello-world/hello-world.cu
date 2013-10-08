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

__global__ void helloKernel(const int* const g_in, int* const g_out, const unsigned int n)
{
  unsigned int global_id = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (global_id < n) {
    g_out[global_id] = g_in[global_id] * g_in[global_id];
  }
}

//nvcc hello.cu -o hello.out
//./hello.out
int main(int argc, char **argv) {

  const int nr_of_elements = 10;

  // initialize input array with dummy data
  int *h_in = new int[nr_of_elements];
  for (int i = 0; i < nr_of_elements; i++) {
    h_in[i] = i;
  }

  // allocate and copy input data to device
  int *d_in;
  checkCudaErrors(cudaMalloc(&d_in, 2 * sizeof(int) * nr_of_elements));
  checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(int) * nr_of_elements, cudaMemcpyHostToDevice));

  // allocate array for output
  int *h_out = new int[nr_of_elements];
  int *d_out;
  checkCudaErrors(cudaMalloc(&d_out, sizeof(int) * nr_of_elements));

  // dimensions of grid and blocks
  const dim3 blockSize(128);
  const dim3 gridSize(256);

  // run kernel
  helloKernel<<<gridSize, blockSize>>>(d_in, d_out, nr_of_elements);

  // copy output from device to host
  checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(int) * nr_of_elements, cudaMemcpyDeviceToHost));

  // print output array
  for(int i = 0; i < nr_of_elements; i++) {
    std::cout << h_out[i] << ",";
  }
  std::cout << std::endl;
}
