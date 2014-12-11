#include <cstdio>
#include <cstdlib>
#include <windows.h>
#include <ctime>


#define WIDTH		(4 * 1024)	
#define	TILE_WIDTH	16		// block will be (TILE_WIDTH,TILEWIDTH)
#define	GRID_WIDTH	(WIDTH / TILE_WIDTH)	// grid will be (GRID_WDITH,GRID_WDITH)


void genData(float* ptr, unsigned int size) {
	while (size--) {
		*ptr++ = (float)(rand() % 1000) / 1000.0F;
	}
}


__global__ void matmul(float* c, const float* a, const float* b, const int width) {
	
	__shared__ float s_a[TILE_WIDTH][TILE_WIDTH];
	__shared__ float s_b[TILE_WIDTH][TILE_WIDTH];
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	register float sum = 0.0F;
	for (register int p = 0; p < width / TILE_WIDTH; ++p) {
		s_a[ty][tx] = a[y * width + (p * TILE_WIDTH + tx)];
		s_b[ty][tx] = b[(p * TILE_WIDTH + ty) * width + x];
		__syncthreads();
		for (register int k = 0; k < TILE_WIDTH; ++k) {
			sum += s_a[ty][k] * s_b[k][tx];
		}
		__syncthreads();
	}
	c[y * width + x] = sum;
}


int main(void) {
	float* pA = NULL;
	float* pB = NULL;
	float* pC = NULL;
	long long cntStart, cntEnd, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)(&freq));
	// malloc memories on the host-side
	pA = (float*)malloc(WIDTH * WIDTH * sizeof(float));
	pB = (float*)malloc(WIDTH * WIDTH * sizeof(float));
	pC = (float*)malloc(WIDTH * WIDTH * sizeof(float));
	printf("pA, pB, pC = %#x %#x %#x\n", pA, pB, pC);
	// generate source data
	genData(pA, WIDTH * WIDTH);
	genData(pB, WIDTH * WIDTH);
	// CUDA: allocate device memory
	float* pAdev = NULL;
	float* pBdev = NULL;
	float* pCdev = NULL;
	cudaMalloc((void**)&pAdev, WIDTH * WIDTH * sizeof(float));
	cudaMalloc((void**)&pBdev, WIDTH * WIDTH * sizeof(float));
	cudaMalloc((void**)&pCdev, WIDTH * WIDTH * sizeof(float));
	printf("pAdev, pBdev, pCdev = %#x %#x %#x\n", pAdev, pBdev, pCdev);
	// CUDA: copy from host to device
	cudaMemcpy(pAdev, pA, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pBdev, pB, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice);
	// start the timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart)); // start the stop watch
	// CUDA: launch the kernel
	dim3 dimGrid(GRID_WIDTH, GRID_WIDTH, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	matmul << < dimGrid, dimBlock >> >(pCdev, pAdev, pBdev, WIDTH);
	cudaPeekAtLastError();
	// end the timer
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd)); // end the stop watch
	printf("elapsed time = %f msec\n", (double)(cntEnd - cntStart) * 1000.0 / (double)(freq));
	// CUDA: copy from device to host
	cudaMemcpy(pC, pCdev, WIDTH * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);
	// print sample cases
	int i, j;
	i = 0; j = 0; printf("c[%4d][%4d] = %f\n", i, j, pC[i * WIDTH + j]);
	i = WIDTH / 2; j = WIDTH / 2; printf("c[%4d][%4d] = %f\n", i, j, pC[i * WIDTH + j]);
	i = WIDTH - 1; j = WIDTH - 1; printf("c[%4d][%4d] = %f\n", i, j, pC[i * WIDTH + j]);
	cudaFree(pAdev);
	cudaFree(pBdev);
	cudaFree(pCdev);
	free(pA);
	free(pB);
	free(pC);
	system("pause");
}

