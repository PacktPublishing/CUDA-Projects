#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>		// --- ifstream
#include <sstream>		// --- stringstream
#include <string>
#include <iostream>		// --- cout

#include <cuda_runtime.h>

texture<unsigned char, 2, cudaReadModeNormalizedFloat> texReference;

float transl_x = 80.0f, transl_y = 80.0f;		// --- Image translation
float scaleFactor = 1.0f / 8.0f;		// --- Image scaleFactor
//float transl_x = 0.0f, transl_y = 0.0f;		// --- Image translation
//float scaleFactor = 2.f;		// --- Image scaleFactor
float xCenter, yCenter;			// --- Image centre
										
unsigned int imageWidth, imageHeight;
dim3 blockSize(16, 16);

/******************/
/* ERROR CHECKING */
/******************/
#define cudaCHECK(ans) { checkAssert((ans), __FILE__, __LINE__); }
inline void checkAssert(cudaError_t errorCode, const char *file, int line, bool abort = true) {
	if (errorCode != cudaSuccess) {
		fprintf(stderr, "Check assert: %s %s %d\n", cudaGetErrorString(errorCode), file, line);
		if (abort) exit(errorCode);
	}
}

cudaArray *d_imageArray = 0;

/**************************/
/* TEXTURE INITIALIZATION */
/**************************/
void initTexture(int imageWidth, int imageHeight, unsigned char *h_samples)
{
	// --- Allocate CUDA array and copy image data
	cudaChannelFormatDesc channelDescrptr = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaCHECK(cudaMallocArray(&d_imageArray, &channelDescrptr, imageWidth, imageHeight));
	unsigned int sz = imageWidth * imageHeight * sizeof(unsigned char);
	cudaCHECK(cudaMemcpyToArray(d_imageArray, 0, 0, h_samples, sz, cudaMemcpyHostToDevice));
	free(h_samples);

	// --- Texture set up
	texReference.addressMode[0] = cudaAddressModeClamp;
	//texReference.addressMode[1] = cudaAddressModeBorder;
	texReference.addressMode[1] = cudaAddressModeClamp;
	
	texReference.filterMode = cudaFilterModeLinear;
	texReference.normalized = false;    

	// --- Texture binding
	cudaCHECK(cudaBindTextureToArray(texReference, d_imageArray));
}

/******************/
/* LOAD PGM IMAGE */
/******************/
void loadPGMImageAndInitTexture(const char *inputFilename) {

	std::cout << "Opening file " << inputFilename << std::endl;
	std::ifstream infile(inputFilename, std::ios::binary);
	std::stringstream ss;
	std::string inputLine = "";

	// --- Read the first line
	getline(infile, inputLine);
	if (inputLine.compare("P5") != 0) std::cerr << "Version error" << std::endl;
	std::cout << "Version : " << inputLine << std::endl;

	std::string identifier;
	std::stringstream::pos_type pos = ss.tellg();
	ss >> identifier;
	if (identifier == "#") {
		// --- If second line is a comment, display the comment
		std::cout << "Comment: " << identifier << std::endl;
	}
	else {
		// --- If second line is not a comment, rewind
		ss.clear();
		ss.seekg(pos, ss.beg);
	}

	// --- Read the third line : width, height
	ss << infile.rdbuf(); 
	ss >> imageWidth >> imageHeight;
	std::cout << "Image size is " << imageWidth << " columns and " << imageHeight << " rows" << std::endl;
	// --- Maximum intensity value
	int max_val;
	ss >> max_val;
	std::cout << "Image maximum intensity is " << max_val << std::endl;

	unsigned char pixel;
	unsigned char *h_samples = (unsigned char *)malloc(imageHeight * imageWidth * sizeof(unsigned char));

	for (int row = 0; row < imageHeight; row++) {//record the pixel values
		for (int col = 0; col < imageWidth; col++) {
			ss.read((char *)&pixel, 1);
			h_samples[row * imageWidth + col] = pixel;
		}
	}

	xCenter = imageWidth * 0.5f;
	yCenter = imageHeight * 0.5f;

	// --- Texture initialization
	initTexture(imageWidth, imageHeight, h_samples);

}

/*******************/
/* WRITE PGM IMAGE */
/*******************/
void writePGMImage(const char *outputFilename, unsigned char *h_samples, const int imageWidth, const int imageHeight) {

	std::ofstream f(outputFilename, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);

	int maxColorValue = 255;
	f << "P5\n" << imageWidth << " " << imageHeight << "\n" << maxColorValue << "\n";

	for (int i = 0; i < imageHeight; ++i)
		f.write(reinterpret_cast<const char*>(&h_samples[i * imageWidth]), imageWidth);

	//if (wannaFlush)
		f << std::flush;
}

/*******************/
/* iDivUp FUNCTION */
/*******************/
int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

/************************/
/* NERP AND LERP KERNEL */
/************************/
template <typename inType>
__host__ __device__ inline inType lerpDevice(inType v0, inType v1, inType t) { return (1 - t) * v0 + t * v1; }

template<class inType, class outType> __device__ outType bilinearDeviceLookUp(const texture<inType, 2, cudaReadModeNormalizedFloat> texReference, float xCoord, float yCoord)
{
	xCoord += 0.5f;
	yCoord += 0.5f;
	float p_x = floor(xCoord);  
	float p_y = floor(yCoord);
	float f_x = xCoord - p_x;     
	float f_y = yCoord - p_y;

	return lerpDevice(lerpDevice(tex2D(texReference, p_x, p_y), tex2D(texReference, p_x + 1.0f, p_y), f_x), lerpDevice(tex2D(texReference, p_x, p_y + 1.0f), tex2D(texReference, p_x + 1.0f, p_y + 1.0f), f_x), f_y);
}

__global__ void nalKernel(unsigned char * __restrict__ d_interpSamples, const unsigned int imWidth, const unsigned int imHeight, const float transl_x, const float transl_y,
	const float scaleFactor, const float xCenter, const float yCenter)
{
	const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	float xNew = (tidx - xCenter) * scaleFactor + xCenter + transl_x;
	float yNew = (tidy - yCenter) * scaleFactor + yCenter + transl_y;

	if ((tidx < imWidth) && (tidy < imHeight))
	{
		// --- Use this line for nearest neighbor or bilinear interpolations using texture filtering
		//float outSample = tex2D(texReference, xNew + 0.5f, yNew + 0.5f);
		// --- Use this line for bilinear using texture lookup
		float outSample = bilinearDeviceLookUp<unsigned char, float>(texReference, xNew, yNew);
		d_interpSamples[tidy * imWidth + tidx] = outSample * 0xff;
	}
}

/*******************************/
/* BICUBIC WITH TEXTURE LOOKUP */
/*******************************/
template<class inOutType> __device__ inOutType cubicInterp(float xCoord, inOutType fm_minus_1, inOutType fm, inOutType fm_plus_1, inOutType fm_plus_2)
{
	inOutType outR = fm_minus_1 * w_0(xCoord) + fm * w_1(xCoord) + fm_plus_1 * w_2(xCoord) + fm_plus_2 * w_3(xCoord);
	return outR;
}

template<class inType, class outType>
__device__ outType bicubicDeviceLookUp(const texture<inType, 2, cudaReadModeNormalizedFloat> texReference, float xCoord, float yCoord)
{
	xCoord += 0.5f;
	yCoord += 0.5f;
	float Indm = floor(xCoord);
	float Indn = floor(yCoord);
	float alpha_x = xCoord - Indm;
	float alpha_y = yCoord - Indn;

	outType f_x_n_minus_1   = cubicInterp<outType>(alpha_x, tex2D(texReference, Indm - 1, Indn - 1), tex2D(texReference, Indm, Indn - 1), tex2D(texReference, Indm + 1, Indn - 1), tex2D(texReference, Indm + 2, Indn - 1));
	outType f_x_n			= cubicInterp<outType>(alpha_x, tex2D(texReference, Indm - 1, Indn),     tex2D(texReference, Indm, Indn),     tex2D(texReference, Indm + 1, Indn),     tex2D(texReference, Indm + 2, Indn));
	outType f_x_n_plus_1	= cubicInterp<outType>(alpha_x, tex2D(texReference, Indm - 1, Indn + 1), tex2D(texReference, Indm, Indn + 1), tex2D(texReference, Indm + 1, Indn + 1), tex2D(texReference, Indm + 2, Indn + 1));
	outType f_x_n_plus_2	= cubicInterp<outType>(alpha_x, tex2D(texReference, Indm - 1, Indn + 2), tex2D(texReference, Indm, Indn + 2), tex2D(texReference, Indm + 1, Indn + 2), tex2D(texReference, Indm + 2, Indn + 2));

	return cubicInterp<outType>(alpha_y, f_x_n_minus_1, f_x_n, f_x_n_plus_1, f_x_n_plus_2);
}

__global__ void bicubicKernelLookUp(unsigned char *d_interpSamples, const unsigned int imWidth, const unsigned int imHeight, const float transl_x, const float transl_y, const float scaleFactor, 
	const float xCenter, const float yCenter) {
	
	const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	float xNew = (tidx - xCenter) * scaleFactor + xCenter + transl_x;
	float yNew = (tidy - yCenter) * scaleFactor + yCenter + transl_y;

	if ((tidx < imWidth) && (tidy < imHeight))
	{
		float outSample = bicubicDeviceLookUp<unsigned char, float>(texReference, xNew, yNew);
		d_interpSamples[tidy * imWidth + tidx] = outSample * 0xff;
	}
}

/**************************************/
/* BICUBIC WITH TEXTURE INTERPOLATION */
/**************************************/
__device__ float w_0(float alpha) { return (1.0f / 6.0f) * (alpha * (alpha * (-alpha + 3.0f) - 3.0f) + 1.0f); }
__device__ float w_1(float alpha) { return (1.0f / 6.0f) * (alpha * alpha * (3.0f * alpha - 6.0f) + 4.0f); }
__device__ float w_2(float alpha) { return (1.0f / 6.0f) * (alpha * (alpha * (-3.0f * alpha + 3.0f) + 3.0f) + 1.0f); }
__device__ float w_3(float alpha) { return (1.0f / 6.0f) * (alpha * alpha * alpha); }
__device__ float g_0(float alpha) { return w_0(alpha) + w_1(alpha); }
__device__ float g_1(float alpha) { return w_2(alpha) + w_3(alpha); }
__device__ float h_0(float alpha) { return -1.0f + w_1(alpha) / (w_0(alpha) + w_1(alpha)) + 0.5f; }
__device__ float h_1(float alpha) { return 1.0f + w_3(alpha) / (w_2(alpha) + w_3(alpha)) + 0.5f; }

template<class inType, class outType>  
__device__ outType bicubicDeviceFiltering(const texture<inType, 2, cudaReadModeNormalizedFloat> texReference, float xCoord, float yCoord)
{
	xCoord += 0.5f;
	yCoord += 0.5f;
	float p_x = floor(xCoord);
	float p_y = floor(yCoord);
	float f_x = xCoord - p_x;
	float f_y = yCoord - p_y;

	float g0_x = g_0(f_x);
	float g1_x = g_1(f_x);
	float h0_x = h_0(f_x);
	float h1x = h_1(f_x);
	float h0_y = h_0(f_y);
	float h1_y = h_1(f_y);

	outType outR = g_0(f_y) * (g0_x * tex2D(texReference, p_x + h0_x, p_y + h0_y) + g1_x * tex2D(texReference, p_x + h1x, p_y + h0_y)) +
		        g_1(f_y) * (g0_x * tex2D(texReference, p_x + h0_x, p_y + h1_y) + g1_x * tex2D(texReference, p_x + h1x, p_y + h1_y));
	return outR;
}

__global__ void bicubicKernelFiltering(unsigned char *d_interpSamples, const unsigned int imWidth, const unsigned int imHeight, const float transl_x, const float transl_y, const float scaleFactor, 
	const float xCenter, const float yCenter) {
	
	const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	float xNew = (tidx - xCenter) * scaleFactor + xCenter + transl_x;
	float yNew = (tidy - yCenter) * scaleFactor + yCenter + transl_y;

	if ((tidx < imWidth) && (tidy < imHeight))
	{
		float outSample = bicubicDeviceFiltering<unsigned char, float>(texReference, xNew, yNew);
		d_interpSamples[tidy * imWidth + tidx] = outSample * 0xff;
	}
}

/**********************/
/* CATMULL-ROM SPLINE */
/**********************/
__device__ float catRom_w_0(float a) { return a * (-0.5f + a * (1.0f - 0.5f * a)); }
__device__ float catRom_w_1(float a) { return 1.0f + a * a * (-2.5f + 1.5f * a); }
__device__ float catRom_w_2(float a) { return a * (0.5f + a * (2.0f - 1.5f * a)); }
__device__ float catRom_w_3(float a) { return a * a * (-0.5f + 0.5f * a); }

template<class inType> __device__ inType catRomInterp(float x, inType fm_minus_1, inType fm, inType fm_plus_1, inType fm_plus_2) 
{	
	inType outR = fm_minus_1 * catRom_w_0(x) + fm * catRom_w_1(x) + fm_plus_1 * catRom_w_2(x) + fm_plus_2 * catRom_w_3(x);
	return outR;
}

template<class inType, class outType>  
__device__ outType catRomDeviceLookUp(const texture<inType, 2, cudaReadModeNormalizedFloat> texReference, float xCoord, float yCoord)
{
	xCoord += 0.5f;
	yCoord += 0.5f;
	float Indm = floor(xCoord);
	float Indn = floor(yCoord);
	float alpha_x = xCoord - Indm;
	float alpha_y = yCoord - Indn;

	outType f_x_n_minus_1   = cubicInterp<outType>(alpha_x, tex2D(texReference, Indm - 1, Indn - 1), tex2D(texReference, Indm, Indn - 1), tex2D(texReference, Indm + 1, Indn - 1), tex2D(texReference, Indm + 2, Indn - 1));
	outType f_x_n			= cubicInterp<outType>(alpha_x, tex2D(texReference, Indm - 1, Indn),     tex2D(texReference, Indm, Indn),     tex2D(texReference, Indm + 1, Indn), tex2D(texReference, Indm + 2, Indn));
	outType f_x_n_plus_1	= cubicInterp<outType>(alpha_x, tex2D(texReference, Indm - 1, Indn + 1), tex2D(texReference, Indm, Indn + 1), tex2D(texReference, Indm + 1, Indn + 1), tex2D(texReference, Indm + 2, Indn + 1));
	outType f_x_n_plus_2	= cubicInterp<outType>(alpha_x, tex2D(texReference, Indm - 1, Indn + 2), tex2D(texReference, Indm, Indn + 2), tex2D(texReference, Indm + 1, Indn + 2), tex2D(texReference, Indm + 2, Indn + 2));

	return catRomInterp<outType>(alpha_y, f_x_n_minus_1, f_x_n, f_x_n_plus_1, f_x_n_plus_2);

}

__global__ void catRomKernelLookUp(unsigned char *d_interpSamples, const unsigned int imWidth, const unsigned int imHeight, const float transl_x, const float transl_y, const float scaleFactor,
	const float xCenter, const float yCenter) {

	const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tidy = blockIdx.y * blockDim.y + threadIdx.y;

	float xNew = (tidx - xCenter) * scaleFactor + xCenter + transl_x;
	float yNew = (tidy - yCenter) * scaleFactor + yCenter + transl_y;

	if ((tidx < imWidth) && (tidy < imHeight))
	{
		float outSample = catRomDeviceLookUp<unsigned char, float>(texReference, xNew, yNew);
		d_interpSamples[tidy * imWidth + tidx] = outSample * 0xff;
	}
}

/********/
/* MAIN */
/********/
int main()
{
	char *inputFilename  = "D:\\Project\\Packt\\ChapI\\Codici\\bicubicTexture\\data\\basilicaJulia.pgm";
	char *outputFilename = "D:\\Project\\Packt\\ChapI\\Codici\\bicubicTexture\\data\\Fig14.pgm";

	loadPGMImageAndInitTexture(inputFilename);

	unsigned char *d_interpSamples; cudaCHECK(cudaMalloc(&d_interpSamples, imageWidth * imageHeight * sizeof(unsigned char)));
	unsigned char *h_result = (unsigned char *)malloc(imageWidth * imageHeight * sizeof(unsigned char));

	dim3 gridSize(imageWidth / blockSize.x, imageHeight / blockSize.y);

	//// --- Nearest neighbor and bilinear filterings
	//// --- Use next line for nearest neighbor using texture filtering 
	//texReference.filterMode = cudaFilterModePoint;
	//// --- Use next line for bilinear using texture filtering 
	////texReference.filterMode = cudaFilterModeLinear;
	//nalKernel << <gridSize, blockSize >> >(d_interpSamples, imageWidth, imageHeight, transl_x, transl_y, scaleFactor, xCenter, yCenter);
	//cudaCHECK(cudaPeekAtLastError());
	//cudaCHECK(cudaDeviceSynchronize());

	// --- Bicubic with texture filtering
	//texReference.filterMode = cudaFilterModeLinear;
	//bicubicKernelFiltering << <gridSize, blockSize >> >(d_interpSamples, imageWidth, imageHeight, transl_x, transl_y, scaleFactor, xCenter, yCenter);
	//cudaCHECK(cudaPeekAtLastError());
	//cudaCHECK(cudaDeviceSynchronize());

	// --- Bicubic with texture lookup
	texReference.filterMode = cudaFilterModePoint;
	bicubicKernelLookUp << <gridSize, blockSize >> >(d_interpSamples, imageWidth, imageHeight, transl_x, transl_y, scaleFactor, xCenter, yCenter);
	cudaCHECK(cudaPeekAtLastError());
	cudaCHECK(cudaDeviceSynchronize());

	// --- Catmul-Rom
	//texReference.filterMode = cudaFilterModePoint;
	//catRomKernelLookUp << <gridSize, blockSize >> >(d_interpSamples, imageWidth, imageHeight, transl_x, transl_y, scaleFactor, xCenter, yCenter);
	//cudaCHECK(cudaPeekAtLastError());
	//cudaCHECK(cudaDeviceSynchronize());

	cudaCHECK(cudaMemcpy(h_result, d_interpSamples, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	writePGMImage(outputFilename, h_result, imageWidth, imageHeight);

	free(h_result);

	return 0;
}
