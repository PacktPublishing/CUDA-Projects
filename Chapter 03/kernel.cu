#include <opencv2\opencv.hpp>

#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 32

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/****************/
/* PRINT KERNEL */
/****************/
__global__ void printKernel(const float * __restrict__ srcPtr, const size_t srcStep, const int Nrows, const int Ncols) {

	int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
	int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (rowIdx >= Nrows || colIdx >= Ncols) return;

	float *rowSrcPtr = (float *)(((char *)srcPtr) + rowIdx * srcStep);

	printf("%d %d %d %f\n", srcStep, rowIdx, colIdx, rowSrcPtr[colIdx]);

}

/********/
/* MAIN */
/********/
int main() {

	// --- 2x3 float, one-channel matrix
	cv::Mat h_A(2, 3, CV_32FC1);

	// --- First row
	h_A.at<float>(0, 0) = 3.f;
	h_A.at<float>(0, 1) = 4.f;
	h_A.at<float>(0, 2) = 12.f;
	// --- Second row
	h_A.at<float>(1, 0) = -1.f;
	h_A.at<float>(1, 1) = 0.3f;
	h_A.at<float>(1, 2) = -4.323f;

	std::cout << "Matrix = " << std::endl << " " << h_A << std::endl << std::endl;

	printf("\n\nTotal number of elements of A is %d\n", h_A.total());

	printf("Matrix allocation continuous? %d\n", h_A.isContinuous());
	
	printf("\n\nCPU memory organization - version 1\n");
	for (int r = 0; r < h_A.rows; r++) {
		float *rowPointer = h_A.ptr<float>(r);
		for (int c = 0; c < h_A.cols; c++) {
			printf("%f\n", rowPointer[c]);
		}
	}

	printf("\n\nCPU memory organization - version 2\n");
	float *dataPointer = (float *)h_A.data;
	for (int k = 0; k < h_A.total(); k++) {
		printf("%f\n", dataPointer[k]);
	}

	cv::cuda::GpuMat d_A(h_A);
	//cv::cuda::GpuMat d_A;
	//d_A.upload(h_A);
	
	printf("\n\nGpuMat continuous = %d\n", d_A.isContinuous());

	std::string ty = cv::typeToString(d_A.type());
	printf("\n\nGpuMat image is %s %dx%d \n", ty.c_str(), d_A.rows, d_A.cols);
	
	float *h_Atest = (float *)malloc(d_A.rows * d_A.cols * sizeof(float));
	gpuErrchk(cudaMemcpy2D(h_Atest, 
		d_A.cols * sizeof(float), 
		(float *)d_A.data,
		d_A.step,
		d_A.cols * sizeof(float),
		d_A.rows,
		cudaMemcpyDeviceToHost));

	printf("\n\nUsing cudaMemcpy2D\n");
	for (int k = 0; k < d_A.rows * d_A.cols; k++) {
		printf("%f\n", h_Atest[k]);
	}

	printf("\n\nUsing kernel call\n");
	dim3 blockDim(BLOCKSIZE_X, BLOCKSIZE_Y);
	dim3 gridDim(1, 1);

	printKernel<<<gridDim, blockDim>>>((float *)d_A.data, d_A.step, d_A.rows, d_A.cols);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	printf("\n\nUsing cudaMemcpy\n");
	for (int r = 0; r < d_A.rows; r++) {
		float *rowPointer = d_A.ptr<float>(r);
		gpuErrchk(cudaMemcpy(h_Atest, rowPointer, d_A.cols * sizeof(float), cudaMemcpyDeviceToHost));
		for (int c = 0; c < d_A.cols; c++) {
			printf("%f\n", h_Atest[c]);
		}
	}

	return 0;
}
