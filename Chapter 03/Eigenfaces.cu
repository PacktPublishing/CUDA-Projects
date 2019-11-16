#include <opencv2\opencv.hpp>

#include <cusolverDn.h>

//#define SHOW_DATASET
//#define SHOW_AVERAGE
//#define SHOW_SHIFTED_IMAGES

#define BLOCKSIZE_X	16
#define BLOCKSIZE_Y	16

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
	case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:	return "Matrix type not supported";
	}

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

/*******************/
/* iDivUp FUNCTION */
/*******************/
int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

/**********************/
/* REMOVE MEAN KERNEL */
/**********************/
__global__ void removeMeanKernel(const float * __restrict__ srcPtr, float * __restrict__ dstPtr, const size_t srcStep, const size_t dstStep, const int Nrows, const int Ncols) {

	int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
	int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (rowIdx >= Nrows || colIdx >= Ncols) return;

	const float *rowSrcPtr = (const float *)(((char *)srcPtr) + 0 * srcStep);
	float *rowDstPtr = (float *)(((char *)dstPtr) + rowIdx * dstStep);

	rowDstPtr[colIdx] = rowDstPtr[colIdx] - rowSrcPtr[colIdx];
}

/********/
/* MAIN */
/********/
int main() {

	// --- Customized
	const int numImages = 24;
	const int nRows = 64;
	const int nCols = 64;
	const int numTrain = 24;
	const int cRows = 10;
	const int cCols = 10;
	const int numEigenfaces = 7;

	/***********/
	/* STEP #1 */
	/***********/
	// --- Image path
	std::string pathToData("D:\\Project\\Packt\\Eigenfaces\\Customized\\");

	// --- GPU memory allocation
	cv::cuda::GpuMat d_A(numTrain, nRows * nCols, CV_32FC1);

	// --- Loading dataset images
	cv::Mat h_imageTemp, h_imageTempCast, h_imageTempResized;
	float *rowPointer;
	for (int k = 0; k < numImages; k++) {

		std::stringstream ss;
		ss << std::setw(3) << std::setfill('0') << k;
		std::string s = ss.str();

		h_imageTemp = cv::imread(pathToData + s + ".png", -1);
		cv::transpose(h_imageTemp, h_imageTemp);
		h_imageTemp.convertTo(h_imageTempCast, CV_32FC1);
		h_imageTempCast = h_imageTempCast.reshape(0, 1);

		std::string ty = cv::typeToString(h_imageTemp.type());
		printf("Loaded image: %s %dx%d \n", ty.c_str(), h_imageTemp.rows, h_imageTemp.cols);
		ty = cv::typeToString(h_imageTempCast.type());
		printf("Cast image: %s %dx%d \n", ty.c_str(), h_imageTempCast.rows, h_imageTempCast.cols);
#ifdef SHOW_DATASET
		cv::resize(h_imageTemp, h_imageTempResized, cv::Size(nRows * cRows, nCols * cCols), cv::INTER_CUBIC);
		cv::imshow("Dataset image", h_imageTempResized);
		cv::waitKey(0);
#endif

		// --- Copy generic row
		rowPointer = d_A.ptr<float>(k);
		cudaCHECK(cudaMemcpy2D(rowPointer,
			d_A.step * sizeof(float),
			h_imageTempCast.ptr<float>(0),
			h_imageTempCast.step * sizeof(float),
			h_imageTempCast.cols * sizeof(float),
			1,
			cudaMemcpyHostToDevice));

	}

	/***********/
	/* STEP #2 */
	/***********/
	// --- Average
	cv::cuda::GpuMat d_mean(1, nRows * nCols, CV_32FC1);
	cv::cuda::reduce(d_A, d_mean, 0, 1);
#ifdef SHOW_AVERAGE
	cv::Mat h_mean(d_mean);
	h_mean = h_mean.reshape(0, nRows);
	h_mean.convertTo(h_mean, CV_8UC1);
	cv::resize(h_mean, h_mean, cv::Size(cRows * nRows, cCols * nCols), cv::INTER_CUBIC);
	std::string ty = cv::typeToString(h_mean.type());
	printf("Average image: %s %dx%d \n", ty.c_str(), h_mean.rows, h_mean.cols);
	cv::imshow("Average image", h_mean);
	cv::waitKey(0);
#endif

	/***********/
	/* STEP #3 */
	/***********/
	// --- Shift images
	dim3 blockDim(BLOCKSIZE_X, BLOCKSIZE_Y);
	dim3 gridDim(iDivUp(d_A.cols, BLOCKSIZE_X), iDivUp(d_A.rows, BLOCKSIZE_Y));
	removeMeanKernel << <gridDim, blockDim >> > ((float *)d_mean.data, (float *)d_A.data, d_mean.step, d_A.step, numTrain, nRows * nCols);
	cudaCHECK(cudaPeekAtLastError());
	cudaCHECK(cudaDeviceSynchronize());


#ifdef SHOW_SHIFTED_IMAGES
	count = 0;
	for (int k = 0; k < numImages; k++) {

		cv::Mat h_shiftedImage(1, nRows * nCols, CV_32FC1);

		// --- Copy generic row
		rowPointer = d_A.ptr<float>(count);
		cudaCHECK(cudaMemcpy2D(h_shiftedImage.ptr<float>(0),
			h_shiftedImage.step * sizeof(float),
			rowPointer,
			d_A.step * sizeof(float),
			d_A.cols * sizeof(float),
			1,
			cudaMemcpyDeviceToHost));

		h_shiftedImage = h_shiftedImage.reshape(0, nRows);
		h_shiftedImage.convertTo(h_shiftedImage, CV_8UC1);
		std::string ty = cv::typeToString(h_shiftedImage.type());
		printf("Removed mean image: %s %dx%d \n", ty.c_str(), h_shiftedImage.rows, h_shiftedImage.cols);
		cv::resize(h_shiftedImage, h_shiftedImage, cv::Size(nRows * cRows, nCols * cCols), cv::INTER_CUBIC);
		cv::imshow("Dataset image", h_shiftedImage);
		cv::waitKey(0);

		count++;
	}
#endif

	/***********/
	/* STEP #4 */
	/***********/
	cv::cuda::GpuMat d_A2;
	d_A.copyTo(d_A2);
	cv::cuda::multiply(d_A, 1.f / sqrt((float)numTrain), d_A);

	// --- Allocating SVD space on the device
	float *d_U;			cudaCHECK(cudaMalloc(&d_U, d_A.cols * d_A.cols * sizeof(float)));
	float *d_V;			cudaCHECK(cudaMalloc(&d_V, d_A.rows * d_A.rows * sizeof(float)));
	float *d_S;			cudaCHECK(cudaMalloc(&d_S, min(d_A.rows, d_A.cols) * sizeof(float)));

	float *rWork;	cudaCHECK(cudaMalloc(&rWork, 1 * sizeof(float)));
	int *d_devInfo;	cudaCHECK(cudaMalloc(&d_devInfo, sizeof(int)));

	// --- Compute SVD
	cusolverDnHandle_t cuSolverHandle;
	cuSolverCHECK(cusolverDnCreate(&cuSolverHandle));

	int workSize = 0;
	cuSolverCHECK(cusolverDnSgesvd_bufferSize(cuSolverHandle, d_A.cols, d_A.rows, &workSize));
	float *workArray;	cudaCHECK(cudaMalloc(&workArray, workSize * sizeof(float)));

	cuSolverCHECK(cusolverDnSgesvd(
		cuSolverHandle,
		'A',
		'A',
		d_A.cols,
		d_A.rows,
		(float *)d_A.data,
		d_A.step1(),
		d_S,
		d_U,
		d_A.cols,
		d_V,
		d_A.rows,
		workArray,
		workSize,
		rWork,
		d_devInfo));

	int h_devInfo = 0;
	cudaCHECK(cudaMemcpy(&h_devInfo, d_devInfo, sizeof(int), cudaMemcpyDeviceToHost));

	if (h_devInfo == 0) printf("SVD converged \n");
	else if (h_devInfo < 0) {
		printf("%d-th parameter is wrong \n", -h_devInfo);
		exit(1);
	}
	else {
		printf("WARNING: h_devInfo = %d : SVD did not converge \n", h_devInfo);
	}

	/***********/
	/* STEP #5 */
	/***********/
	cv::cuda::GpuMat d_Umat(d_A.cols, d_A.cols, CV_32FC1, d_U);
	d_Umat(cv::Range(0, numEigenfaces), cv::Range(0, d_Umat.rows)).copyTo(d_Umat);

	/***********/
	/* STEP #6 */
	/***********/
	cv::cuda::GpuMat d_features(numEigenfaces, numTrain, CV_32FC1);
	cv::cuda::gemm(d_Umat, d_A2, 1.f, d_features, 0.f, d_features, cv::GEMM_2_T);

	/***********/
	/* STEP #7 */
	/***********/
	// --- Load test image
	std::stringstream ss;
	ss << std::setw(3) << std::setfill('0') << 19;
	std::string s = ss.str();

	h_imageTemp = cv::imread(pathToData + "seanConneryTestImage.png", -1);
	cv::transpose(h_imageTemp, h_imageTemp);
	h_imageTemp.convertTo(h_imageTempCast, CV_32FC1);
	h_imageTempCast = h_imageTempCast.reshape(0, 1);

	std::string ty = cv::typeToString(h_imageTemp.type());
	//ty = cv::typeToString(h_imageTemp.type());
	printf("Loaded image: %s %dx%d \n", ty.c_str(), h_imageTemp.rows, h_imageTemp.cols);
	ty = cv::typeToString(h_imageTempCast.type());
	printf("Cast image: %s %dx%d \n", ty.c_str(), h_imageTempCast.rows, h_imageTempCast.cols);

	cv::resize(h_imageTemp, h_imageTempResized, cv::Size(nRows * cRows, nCols * cCols), cv::INTER_CUBIC);
	cv::transpose(h_imageTempResized, h_imageTempResized);
	cv::imshow("Test image", h_imageTempResized);
	cv::waitKey(0);

	// --- Copy generic row
	cv::cuda::GpuMat d_testImage(1, nRows * nCols, CV_32FC1);
	rowPointer = d_testImage.ptr<float>(0);
	cudaCHECK(cudaMemcpy2D(rowPointer,
		d_testImage.step * sizeof(float),
		h_imageTempCast.ptr<float>(0),
		h_imageTempCast.step * sizeof(float),
		h_imageTempCast.cols * sizeof(float),
		1,
		cudaMemcpyHostToDevice));

	// --- Subtract the mean database image from the test image
	cv::cuda::subtract(d_testImage, d_mean, d_testImage);

	// --- Compute the feature vector of the test image
	cv::cuda::GpuMat d_featureVec(numEigenfaces, 1, CV_32FC1);
	cv::cuda::gemm(d_Umat, d_testImage, 1.f, d_featureVec, 0.f, d_featureVec, cv::GEMM_2_T);

	/***********/
	/* STEP #8 */
	/***********/
	cv::cuda::GpuMat d_temp(numEigenfaces, 1, CV_32FC1);
	cv::cuda::GpuMat d_similarityScores(numTrain, 1, CV_32FC1);

	for (int t = 0; t < numTrain; t++) {
		d_features(cv::Range(0, numEigenfaces), cv::Range(t, t + 1)).copyTo(d_temp);
		cv::cuda::subtract(d_temp, d_featureVec, d_temp);
		cv::cuda::sqr(d_temp, d_temp);
		cv::cuda::reduce(d_temp, d_similarityScores.row(t), 0, 0);
	}
	
	double minVal, maxVal;
	cv::Point minLoc, maxLoc;
	cv::cuda::minMaxLoc(d_similarityScores, &minVal, &maxVal, &minLoc, &maxLoc);
	std::cout << minVal << " " << maxVal << " " << minLoc << " " << maxLoc << "\n";

	std::stringstream ss2;
	ss2 << std::setw(3) << std::setfill('0') << minLoc.y;
	std::string s2 = ss2.str();

	std::cout << pathToData + s2 + ".png" << std::endl;

	cv::Mat h_recognizedImage, h_recognizedImageResized;
	h_recognizedImage = cv::imread(pathToData + s2 + ".png", -1);
	cv::resize(h_recognizedImage, h_recognizedImageResized, cv::Size(nRows * cRows, nCols * cCols), cv::INTER_CUBIC);
	cv::imshow("Recognized image", h_recognizedImageResized);
	cv::waitKey(0);

}
