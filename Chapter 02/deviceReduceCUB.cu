#include <iostream>

#include <cub\cub.cuh>

#define pi_f  3.14159265358979f                 // Greek pi in single precision

#define BLOCKSIZE		256

/*******************/
/* iDivUp FUNCTION */
/*******************/
int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

/***************************************************/
/* KERNEL FOR COMPUTING THE SEQUENCE TO BE REDUCED */
/***************************************************/
__global__ void preparationKernel(float * __restrict__ d_f, const float a, const float h, const int N) {

	const int tidx = threadIdx.x + blockDim.x * blockIdx.x;

	if (tidx >= N) return;

	float coeff;
	if (tidx & 1 == 1) coeff = 2.f;
	else coeff = 4.f;

	if ((tidx == 0) || (tidx == N - 1)) coeff = 1.f;

	float x = a + tidx * h;

	// --- Quadrature weights x function samples evaluation
	d_f[tidx] = (h / 3.f) * coeff * sin(2.f * pi_f * x);

}

/********/
/* MAIN */
/********/
int main()
{
	// --- Integration domain
	float a = 0.5f;
	float b = 1.f;

	// --- Number of integration nodes
	const int N = 1024 * 256;

	// --- Generate sampling points
	float h = (b - a) / (float)(N - 1);  // --- The number of discretization intervals is the number of integration nodes minus one

	// --- Device memory allocation of the vector to be reduced and its initialization
	float *d_f; 	CubDebugExit(cudaMalloc(&d_f, N * sizeof(float)));

	// --- Quadrature weights x function samples evaluation
	preparationKernel << <iDivUp(N, BLOCKSIZE), BLOCKSIZE >> >(d_f, a, h, N);
	CubDebugExit(cudaPeekAtLastError());
	CubDebugExit(cudaDeviceSynchronize());

	// --- Allocate storage for reduction result
	float *d_s;  CubDebugExit(cudaMalloc(&d_s, sizeof(float)));

	// --- Provide the number of bytes required for the temporary storage needed by CUB. Allocate space for temporary storage
	size_t temp_storage_bytes = 0;
	float *d_temp_storage = nullptr;
	cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_f, d_s, N);
	CubDebugExit(cudaMalloc(&d_temp_storage, temp_storage_bytes));

	// --- Run reduction
	cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_f, d_s, N);

	// --- Copy results to host
	float h_s;
	CubDebugExit(cudaMemcpy(&h_s, d_s, sizeof(float), cudaMemcpyDeviceToHost));

	printf("The integral is %f\n", h_s);

}
