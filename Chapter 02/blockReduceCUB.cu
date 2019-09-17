#include <iostream>

#include <cub\cub.cuh>

#define pi_f  3.14159265358979f                 // Greek pi in single precision

#define BLOCKSIZE		256

/*******************/
/* iDivUp FUNCTION */
/*******************/
int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

/******************************/
/* CUB BLOCK REDUCTION KERNEL */
/******************************/
__global__ void blockReductionKernel(float * __restrict__ d_s, const float a, const float h, const int N)
{
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;

	if (tidx >= N) return;

	float coeff;
	if (tidx & 1 == 1) coeff = 2.f;
	else coeff = 4.f;

	if ((tidx == 0) || (tidx == N - 1)) coeff = 1.f;

	float x = a + tidx * h;

	// --- Quadrature weights x function samples evaluation
	float val = (h / 3.f) * coeff * sin(2.f * pi_f * x);

	// --- Specify the block reduction algorithm to use
	using BlockReduceT = cub::BlockReduce<float, BLOCKSIZE, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;

	// --- Shared memory
	__shared__ typename BlockReduceT::TempStorage temp_storage;

	// --- Run block reduction
	float blockReduction = BlockReduceT(temp_storage).Sum(val);

	// --- Update result
	if (threadIdx.x == 0) { atomicAdd(d_s, blockReduction); }

	return;
}

/********/
/* MAIN */
/********/
int main() {

	// --- Integration domain
	float a = 0.5f;
	float b = 1.f;

	// --- Number of integration nodes
	const int N = 1024 * 256;

	// --- Generate sampling points
	float h = (b - a) / (float)(N - 1);  // --- The number of discretization intervals is the number of integration nodes minus one
										  
	// --- Allocate storage for reduction result
	float *d_s;  CubDebugExit(cudaMalloc(&d_s, sizeof(float)));

	// --- Run reduction
	blockReductionKernel<< <iDivUp(N, BLOCKSIZE), BLOCKSIZE >> >(d_s, a, h, N);
	CubDebugExit(cudaPeekAtLastError());
	CubDebugExit(cudaDeviceSynchronize());

	// --- Copy results to host
	float h_s;
	CubDebugExit(cudaMemcpy(&h_s, d_s, sizeof(float), cudaMemcpyDeviceToHost));

	printf("The integral is %f\n", h_s);

}
