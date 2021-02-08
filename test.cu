#include <cuda.h>
#include <stdio.h>


float * boats;

void checkError()
{
	cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}
__global__ void normalMAKernel(float * d)
{
    float a = 234.3;
    float b = 4672.323;
    float c = 392053.2345;

    d[0] = 1.0/sqrt(c);
}

__global__ void speedyMAKernel(float * d)
{
    float a = 234.3;
    float b = 4672.323;
    float c = 392053.2345;

    d[0] = __frsqrt_rn(c);
}

int main(int argc, char *args[])
{
    cudaSetDevice(1);
    printf("I ran a\n");
    cudaEvent_t normalStart,normalStop, fastStart,fastStop;
    cudaEventCreate(&normalStart);
    cudaEventCreate(&normalStop);
    cudaEventCreate(&fastStart);
    cudaEventCreate(&fastStop);

    cudaError_t err = cudaMalloc(&boats, 100 * sizeof(float));
    if ( err != cudaSuccess ) return 0;

    cudaEventRecord(normalStart);
    normalMAKernel<<<214748,1>>>(boats);
    speedyMAKernel<<<214748,1>>>(boats);
    checkError();
    cudaDeviceSynchronize();
    cudaEventRecord(normalStop);

    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds,normalStart,normalStop);

    printf("It took %f",milliseconds);

}