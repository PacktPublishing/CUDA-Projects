#include <stdio.h>

texture<float, 1> tex;

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

/*********************/
/* TEXTURE FILTERING */
/*********************/
__global__ void textureFilteringKernelNerp(const float * __restrict__ d_samples, const float * __restrict__ d_xCoord, const int numInSamples)
{
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;

	float nn;
	int   ind = (int)round(d_xCoord[tidx]);
	if (d_xCoord[tidx] < 0)
		nn = d_samples[0];
	else if (d_xCoord[tidx] > numInSamples - 1)
		nn = d_samples[numInSamples - 1];
	else
		nn = d_samples[ind];

	printf("argument = %f; texture = %f; nearest neighbor = %f\n", d_xCoord[tidx], tex1D(tex, (d_xCoord[tidx]) + 0.5), nn);
}

__global__ void textureFilteringKernelLerp(const float * __restrict__ d_samples, const float * __restrict__ d_xCoord, const int numInSamples)
{
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;

	float ll;
	if (d_xCoord[tidx] < 0)
		ll = d_samples[0];
	else if (d_xCoord[tidx] > numInSamples - 1)
		ll = d_samples[numInSamples - 1];
	else {
		int ind = floor(d_xCoord[tidx]);
		float alpha = d_xCoord[tidx] - ind;
		ll = (1.f - alpha) * d_samples[ind] + alpha * d_samples[ind + 1];
	}

	printf("argument = %f; texture = %f; linear = %f\n", d_xCoord[tidx], tex1D(tex, (d_xCoord[tidx]) + 0.5), ll);
}

void textureFiltering(float *h_samples, float *d_samples, float *d_xCoord, int numInSamples, int numOutSamples) {

	cudaArray* d_cudaArrayData = NULL; cudaMallocArray(&d_cudaArrayData, &tex.channelDesc, numInSamples, 1);
	cudaMemcpyToArray(d_cudaArrayData, 0, 0, h_samples, sizeof(float) * numInSamples, cudaMemcpyHostToDevice);
	cudaBindTextureToArray(tex, d_cudaArrayData);

	tex.normalized = false;
	tex.addressMode[0] = cudaAddressModeClamp;
	//tex.addressMode[0] = cudaAddressModeBorder;
	//tex.addressMode[0] = cudaAddressModeWrap;
	//tex.addressMode[0] = cudaAddressModeMirror;

	tex.filterMode = cudaFilterModePoint;
	textureFilteringKernelNerp << <1, numOutSamples >> >(d_samples, d_xCoord, numInSamples);
	cudaCHECK(cudaPeekAtLastError());
	cudaCHECK(cudaDeviceSynchronize());	printf("\n\n");

	tex.filterMode = cudaFilterModeLinear;
	textureFilteringKernelLerp << <1, numOutSamples >> >(d_samples, d_xCoord, numInSamples);
	cudaCHECK(cudaPeekAtLastError());
	cudaCHECK(cudaDeviceSynchronize());
}

/********/
/* MAIN */
/********/
int main()
{
	// --- Number of samples
	int numInSamples = 5;

	// --- Number of interpolated samples
	int numOutSamples = 7;

	// --- Input data on host and device
	float *h_samples = (float*)malloc(numInSamples * sizeof(float));
	for (int ind = 0; ind < numInSamples; ind++) {
		h_samples[ind] = (float)ind / (float)numInSamples;
		printf("index = %d; datum = %f\n", ind, h_samples[ind]);
	}
	printf("\n\n");
	float* d_samples;		cudaCHECK(cudaMalloc(&d_samples, sizeof(float) * numInSamples));
	cudaCHECK(cudaMemcpy(d_samples, h_samples, sizeof(float) * numInSamples, cudaMemcpyHostToDevice));

	// --- Output sampling
	float *h_xCoord = (float *)malloc(numOutSamples * sizeof(float));
	h_xCoord[0] = -0.6f; h_xCoord[1] = -0.1f; h_xCoord[2] = 0.6f; h_xCoord[3] = 1.5f; h_xCoord[4] = 2.1f; h_xCoord[5] = 2.9f; h_xCoord[6] = 4.7f;
	float *d_xCoord;		cudaCHECK(cudaMalloc(&d_xCoord, sizeof(float) * numOutSamples));
	cudaCHECK(cudaMemcpy(d_xCoord, h_xCoord, sizeof(float) * numOutSamples, cudaMemcpyHostToDevice));

	textureFiltering(h_samples, d_samples, d_xCoord, numInSamples, numOutSamples);

	return 0;
}

