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

__global__ void stencilKernel(const int* const g_in, int* const g_out, const unsigned int rows, const unsigned int cols)
{

  const int filterWidth = 3;
  const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                       blockIdx.y * blockDim.y + threadIdx.y);

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= cols || thread_2D_pos.y >= rows) {
    return;
  }

  int output_elem = 0;

  for (int i = 0; i < filterWidth; i++) {
    int elem_x = min(max(thread_2D_pos.x - filterWidth / 2 + i, 0), cols - 1);
    output_elem += g_in[elem_x + thread_2D_pos.y * cols];
  }

  for (int i = 0; i < filterWidth; i++) {
    int elem_y = min(max(thread_2D_pos.y - filterWidth / 2 + i, 0), rows - 1);
    output_elem += g_in[thread_2D_pos.x + elem_y * cols];
  }

  output_elem -= g_in[thread_2D_pos.x + thread_2D_pos.y * cols];
  g_out[thread_2D_pos.x + thread_2D_pos.y * cols] = output_elem;
}

int main(int argc, char **argv) {

  const int cols = 5;
  const int rows = 5;
  const int nr_of_elements = cols * rows;

  // initialize input array with dummy data
  int *h_in = new int[nr_of_elements];
  for (int i = 0; i < nr_of_elements; i++) {
    h_in[i] = i;
  }

  // allocate and copy input data to device
  int *d_in;
  checkCudaErrors(cudaMalloc(&d_in, sizeof(int) * nr_of_elements));
  checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(int) * nr_of_elements, cudaMemcpyHostToDevice));

  // allocate array for output
  int *h_out = new int[nr_of_elements];
  int *d_out;
  checkCudaErrors(cudaMalloc(&d_out, sizeof(int) * nr_of_elements));

  // dimensions of grid and blocks
  const dim3 blockSize(12,8);
  const dim3 gridSize(ceil(cols/ (float) blockSize.x), ceil(rows/ (float) blockSize.y));

  // run kernel
  stencilKernel<<<gridSize, blockSize>>>(d_in, d_out, rows, cols);

  // copy output from device to host
  checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(int) * nr_of_elements, cudaMemcpyDeviceToHost));

  // print output array
  for(int i = 0; i < nr_of_elements; i++) {
    std::cout << h_out[i] << ",";
  }
  std::cout << std::endl;
}
