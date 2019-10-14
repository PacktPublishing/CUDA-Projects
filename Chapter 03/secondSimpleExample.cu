#include <opencv2\opencv.hpp>

#include <cusolverDn.h>

#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 32

/******************/
/* ERROR CHECKING */
/******************/
#define cudaCHECK(ans) { checkAssert((ans), __FILE__, __LINE__); }
inline void checkAssert(cudaError_t errorCode, const char *file, int line, bool abort = true)
{
	if (errorCode != cudaSuccess)
	{
		fprintf(stderr, "Check assert: %s %s %d\n", cudaGetErrorString(errorCode), file, line);
		if (abort) exit(errorCode);
	}
}

/***************************/
/* cuSOLVER ERROR CHECKING */
/***************************/
static const char *_cuSolverReturnErrorString(cusolverStatus_t errorCode)
{
	switch (errorCode) {
		case CUSOLVER_STATUS_SUCCESS:			return "cuSolver successful call";
		case CUSOLVER_STATUS_NOT_INITIALIZED:	return "cuSolver is not initialized";
		case CUSOLVER_STATUS_ALLOC_FAILED:		return "cuSolver internal resource allocation failed";
		case CUSOLVER_STATUS_INVALID_VALUE:		return "cuSolver function has an unsupported value or parameter";
		case CUSOLVER_STATUS_ARCH_MISMATCH:		return "cuSolver function requires an unsupported architecture feature";
		case CUSOLVER_STATUS_EXECUTION_FAILED:	return "cuSolver function failed to execute";
		case CUSOLVER_STATUS_INTERNAL_ERROR:	return "cuSolver internal operation failed";
		case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:	return "Matrix type not supported"; }

	return "<unknown>";
}

inline void __cuSolverCHECK(cusolverStatus_t errorCode, const char *file, const int line)
{
	if (CUSOLVER_STATUS_SUCCESS != errorCode) {
		fprintf(stderr, "cuSolver was unsuccessful in file '%s', line %d; the reported errorCodeor is: %s \nterminating!\n", __FILE__, __LINE__, \
			_cuSolverReturnErrorString(errorCode)); \
			assert(0); \
	}
}

void cuSolverCHECK(cusolverStatus_t errorCode) { __cuSolverCHECK(errorCode, __FILE__, __LINE__); }

/**********************/
/* REMOVE MEAN KERNEL */
/**********************/
__global__ void removeMeanKernel(const float * __restrict__ srcPtr, float * __restrict__ dstPtr, const size_t srcStep, const size_t dstStep, const int Nrows, const int Ncols) {
	
	int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
	int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (rowIdx >= Nrows || colIdx >= Ncols) return;
	
	const float *rowSrcPtr = (const float *)(((char *)srcPtr) + 0      * srcStep);
	      float *rowDstPtr = (      float *)(((char *)dstPtr) + rowIdx * dstStep);

	rowDstPtr[colIdx] = rowDstPtr[colIdx] - rowSrcPtr[colIdx];
}

/********/
/* MAIN */
/********/
int main() {

	// --- 2x3 float, one-channel matrix
	cv::Mat h_A1(2, 3, CV_32FC1);
	cv::Mat h_A2(2, 3, CV_32FC1);

	// --- First row
	h_A1.at<float>(0, 0) = 3.f;
	h_A1.at<float>(0, 1) = 4.f;
	h_A1.at<float>(0, 2) = 12.f;
	// --- Second row
	h_A1.at<float>(1, 0) = -1.f;
	h_A1.at<float>(1, 1) = 0.3f;
	h_A1.at<float>(1, 2) = -4.323f;

	// --- First row
	h_A2.at<float>(0, 0) = -1.f;
	h_A2.at<float>(0, 1) = 2.f;
	h_A2.at<float>(0, 2) = 0.34f;
	// --- Second row
	h_A2.at<float>(1, 0) = -2.32f;
	h_A2.at<float>(1, 1) = 37.f;
	h_A2.at<float>(1, 2) = 11.121f;

	std::cout << "Matrix 1 = " << std::endl << " " << h_A1 << std::endl << std::endl;
	std::cout << "Matrix 2 = " << std::endl << " " << h_A2 << std::endl << std::endl;

	// --- Transform matrix A1 into row
	h_A1 = h_A1.reshape(0, 1);
	// --- Transform matrix A2 into row
	h_A2 = h_A2.reshape(0, 1);

	std::cout << "Matrix 1 = " << std::endl << " " << h_A1 << std::endl << std::endl;
	std::cout << "Matrix 2 = " << std::endl << " " << h_A2 << std::endl << std::endl;

	// --- GPU memory allocation
	cv::cuda::GpuMat d_A(2, h_A1.total(), CV_32FC1);

	// --- Copy first row
	float *rowPointer = d_A.ptr<float>(0);
	cudaCHECK(cudaMemcpy2D(rowPointer,
		d_A.step * sizeof(float),
		h_A1.ptr<float>(0),
		h_A1.step * sizeof(float),
		h_A1.cols * sizeof(float),
		1,
		cudaMemcpyHostToDevice));

	// --- Copy second row
	rowPointer = d_A.ptr<float>(1);
	cudaCHECK(cudaMemcpy2D(rowPointer,
		d_A.step * sizeof(float),
		h_A2.ptr<float>(0),
		h_A2.step * sizeof(float),
		h_A2.cols * sizeof(float),
		1,
		cudaMemcpyHostToDevice));

	cv::Mat h_result(d_A);

	std::cout << "CPU -> GPU memory movement: result matrix = " << std::endl << " " << h_result << std::endl << std::endl;

	// --- Average
	cv::cuda::GpuMat d_mean(1, h_A1.total(), CV_32FC1);
	cv::cuda::reduce(d_A, d_mean, 0, 1);

	cv::Mat h_mean(d_mean);

	std::cout << "Average matrix over columns = " << std::endl << " " << h_mean << std::endl << std::endl;
	
	dim3 blockDim(BLOCKSIZE_X, BLOCKSIZE_Y);
	dim3 gridDim(1, 1);

	removeMeanKernel << <gridDim, blockDim >> > ((float *)d_mean.data, (float *)d_A.data, d_mean.step, d_A.step, 2, h_A1.total());
	cudaCHECK(cudaPeekAtLastError());
	cudaCHECK(cudaDeviceSynchronize());

	cv::Mat h_A(d_A);

	std::cout << "Matrix with removed average = " << std::endl << " " << h_A << std::endl << std::endl;

	// --- Compute covariance matrix
	const int Nrows = 2;
	const int Ncols = 2;
	cv::cuda::GpuMat d_Cov(Nrows, Ncols, CV_32FC1);
	cv::cuda::gemm(d_A, d_A, 1.f, d_Cov, 0.f, d_Cov, cv::GEMM_2_T);

	cv::Mat h_Cov(d_Cov);

	std::cout << "Covariance matrix = " << std::endl << " " << h_Cov << std::endl << std::endl;
	
	// --- Compute SVD
	cusolverDnHandle_t cuSolverHandle;
	cuSolverCHECK(cusolverDnCreate(&cuSolverHandle));

	int workSize = 0;
	cuSolverCHECK(cusolverDnSgesvd_bufferSize(cuSolverHandle, Nrows, Ncols, &workSize));
	float *workArray;	cudaCHECK(cudaMalloc(&workArray, workSize * sizeof(float)));
	
	// --- Allocating SVD space on the host
	float *h_U = (float *)malloc(Nrows * Nrows	   * sizeof(float));
	float *h_V = (float *)malloc(Ncols * Ncols	   * sizeof(float));
	float *h_S = (float *)malloc(min(Nrows, Ncols) * sizeof(float));
	
	// --- Allocating SVD space on the device
	float *d_U;			cudaCHECK(cudaMalloc(&d_U, Nrows * Nrows	 * sizeof(float)));
	float *d_V;			cudaCHECK(cudaMalloc(&d_V, Ncols * Ncols	 * sizeof(float)));
	float *d_S;			cudaCHECK(cudaMalloc(&d_S, min(Nrows, Ncols) * sizeof(float)));

	//float *rWork;	cudaCHECK(cudaMalloc(&rWork, 1 * sizeof(float)));
	int *devInfo;	cudaCHECK(cudaMalloc(&devInfo, sizeof(int)));

	cuSolverCHECK(cusolverDnSgesvd(
		cuSolverHandle,
		'A',
		'A',
		Nrows,
		Ncols,
		(float *)d_Cov.data,
		d_Cov.step1(),
		d_S,
		d_U,
		Nrows,
		d_V,
		Ncols,
		workArray,
		workSize,
		NULL,
		//rWork
		devInfo));

	int devInfo_h = 0;	cudaCHECK(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if (devInfo_h != 0) std::cout << "Unsuccessful SVD execution\n\n";

	// --- Moving the results from device to host
	cudaCHECK(cudaMemcpy(h_S, d_S, min(Nrows, Ncols) * sizeof(float), cudaMemcpyDeviceToHost));
	cudaCHECK(cudaMemcpy(h_U, d_U, Nrows * Nrows     * sizeof(float), cudaMemcpyDeviceToHost));
	cudaCHECK(cudaMemcpy(h_V, d_V, Ncols * Ncols     * sizeof(float), cudaMemcpyDeviceToHost));

	printf("\n\nSingular values = %f %f\n", h_S[0], h_S[1]);
	printf("\n\nFirst column of U = %f %f\n", h_U[0], h_U[1]);
	printf("\n\nSecond column of U = %f %f\n", h_U[2], h_U[3]);
	printf("\n\nFirst column of V = %f %f\n", h_V[0], h_V[1]);
	printf("\n\nSecond column of V = %f %f\n", h_V[2], h_V[3]);

	printf("%d\n", d_Cov.step);

	return 0;
}
