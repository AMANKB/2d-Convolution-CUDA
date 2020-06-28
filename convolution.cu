#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <time.h>

#define BLOCK_SIZE 32
#define WA 512   
#define HA 512     
#define HC 3     
#define WC 3
#define WB (WA - WC + 1)
#define HB (HA - HC + 1)


__global__ void Conv2DKernel(float* A, float* B, float* C, int numARows, int numACols, int numBRows, int numBCols, int numCRows, int numCCols)
{
	int col = blockIdx.x * (BLOCK_SIZE - WC + 1) + threadIdx.x;
	int row = blockIdx.y * (BLOCK_SIZE - WC + 1) + threadIdx.y;
	int row_i = row - WC + 1;
	int col_i = col - WC + 1;

	float tmp = 0;

	__shared__ float shm[BLOCK_SIZE][BLOCK_SIZE];

	if (row_i < WA && row_i >= 0 && col_i < WA && col_i >= 0)
	{
		shm[threadIdx.y][threadIdx.x] = A[col_i * WA + row_i];
	}
	else
	{
		shm[threadIdx.y][threadIdx.x] = 0;
	}

	__syncthreads();

	if (threadIdx.y < (BLOCK_SIZE - WC + 1) && threadIdx.x < (BLOCK_SIZE - WC + 1) && row < (WB - WC + 1) && col < (WB - WC + 1))
	{
		for (int i = 0; i< WC;i++)
			for (int j = 0;j<WC;j++)
				tmp += shm[threadIdx.y + i][threadIdx.x + j] * C[j*WC + i];
		B[col*WB + row] = tmp;
	}
}


__global__ void MaxPoolKernel(int channels, int image_height, int image_width, int pool_height, int pool_width,
		double *global_pointer, double *output_pointer)
{
	__shared__ double shared_pointer[2704];
	int pad_width = pool_width/2;
	int pad_height = pool_height/2;
	int block_x_index = blockDim.x*blockIdx.x;
	int block_y_index = blockDim.y*blockIdx.y;

	int global_offset = blockIdx.z*image_width*image_height;
	int global_x_index;
	int global_y_index;

	int i = -1*pad_height;
	int j = -1*pad_width;
	int shared_mem_index = 0;
	for(int i = threadIdx.y; i < blockDim.y + 2*pad_height; i = i + blockDim.y){
		for(int j = threadIdx.x; j < blockDim.x + 2*pad_width; j = j + blockDim.x){
			int shared_mem_index = i*(blockDim.x+ 2*pad_width) + j;
			global_y_index = block_y_index - pad_height;
			global_x_index = block_x_index - pad_width;
			if(global_x_index < 0 || global_x_index >= image_width || global_y_index < 0 || global_y_index >= image_height){
				shared_pointer[shared_mem_index] = 0;
			}else{
				shared_pointer[shared_mem_index] = global_pointer[global_offset+ (global_y_index*WIDTH) + global_x_index];
			}
		}
	}
	__syncthreads();
	double max_value = 0.0;
	for(int i = 0; i < pool_height; i++){
		for(int j = 0; j < pool_width; j++){
			int loc_index = (i+threadIdx.y)*(blockDim.x + 2*pad_width) + (j+threadIdx.x);
			if(shared_pointer[loc_index] > max_value){
				max_value = shared_pointer[loc_index];
			}
		}
	}
	global_y_index = block_y_index + threadIdx.y;
	global_x_index = block_x_index + threadIdx.x;
	output_pointer[global_offset+ (global_y_index*WIDTH) + global_x_index] = max_value;
}

void randomInit(float* data, int size)
{
	for (int i = 0; i < size; ++i)
		data[i] = rand() / (float)RAND_MAX;
}

int main(int argc, char** argv)
{
	srand(2006);
	cudaError_t error;
	cudaEvent_t start_G, stop_G;

	cudaEventCreate(&start_G);
	cudaEventCreate(&stop_G);

	unsigned int size_A = WA * HA;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float* h_A = (float*)malloc(mem_size_A);

	unsigned int size_B = WB * HB;
	unsigned int mem_size_B = sizeof(float) * size_B;
	float* h_B = (float*)malloc(mem_size_B);

	unsigned int size_C = WC * HC;
	unsigned int mem_size_C = sizeof(float) * size_C;
	float* h_C = (float*)malloc(mem_size_C);

	randomInit(h_A, size_A);
	randomInit(h_C, size_C);

	float* d_A;
	float* d_B;
	float* d_C;

	error = cudaMalloc((void**)&d_A, mem_size_A);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s  in cudaMalloc for A\n", cudaGetErrorString(error));
		return EXIT_FAILURE;
	}

	error = cudaMalloc((void**)&d_B, mem_size_B);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s  in cudaMalloc for B\n", cudaGetErrorString(error));
		return EXIT_FAILURE;
	}

	error = cudaMalloc((void**)&d_C, mem_size_C);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s  in cudaMalloc for C\n", cudaGetErrorString(error));
		return EXIT_FAILURE;
	}


	error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s  in cudaMemcpy for A\n", cudaGetErrorString(error));
		return EXIT_FAILURE;
	}

	error = cudaMemcpy(d_C, h_C, mem_size_C, cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s  in cudaMemcpy for C\n", cudaGetErrorString(error));
		return EXIT_FAILURE;
	}

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((WB - 1) / (BLOCK_SIZE - WC + 1), (WB - 1) / (BLOCK_SIZE - WC + 1));

	Convolution << < grid, threads >> >(d_A, d_B, d_C, HA, WA, HB, WB, HC, WC);

	cudaEventRecord(start_G);

	Convolution << < grid, threads >> >(d_A, d_B, d_C, HA, WA, HB, WB, HC, WC);
	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s  in launching kernel\n", cudaGetErrorString(error));
		return EXIT_FAILURE;
	}

	error = cudaDeviceSynchronize();

	if (error != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s  in cudaDeviceSynchronize \n", cudaGetErrorString(error));
		return EXIT_FAILURE;
	}

	cudaEventRecord(stop_G);

	cudaEventSynchronize(stop_G);

	error = cudaMemcpy(h_B, d_B, mem_size_B, cudaMemcpyDeviceToHost);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s  in cudaMemcpy for B\n", cudaGetErrorString(error));
		return EXIT_FAILURE;
	}


	float miliseconds = 0;
	cudaEventElapsedTime(&miliseconds, start_G, stop_G);

	printf("Time took to compute matrix A of dimensions %d x %d  on GPU is %f ms \n \n \n", WA, HA, miliseconds);

	for (int i = 0;i < HB;i++)
	{
		for (int j = 0;j < WB;j++)
		{
			printf("%f ", h_B[i*HB + j]);
		}
		printf("\n");
	}

	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return EXIT_SUCCESS;
}
