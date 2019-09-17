#include <curand.h>

#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

#include <time.h>

/*************************/
/* CURAND ERROR CHECKING */
/*************************/
static const char *_curandReturnErrorString(curandStatus_t error)
{
	switch (error)
	{
	case 0:   return "No errors";

	case 100: return "Header file and linked library version do not match";

	case 101: return "Generator not initialized";

	case 102: return "Memory allocation failed";

	case 103: return "Generator is wrong type";

	case 104: return "Argument out of range";

	case 105: return "Length requested is not a multiple of dimension";

	case 106: return "GPU does not have double precision required by MRG32k3a";

	case 201: return "Kernel launch failure";

	case 202: return "Preexisting failure on library entry";

	case 203: return "Initialization of CUDA failed";

	case 204: return "Architecture mismatch, GPU does not support requested feature";

	case 999: return "Internal library error";

	}

	return "<unknown>";
}

inline void __curandCHECK(curandStatus_t err, const char *file, const int line)
{
	if (CURAND_STATUS_SUCCESS != err) {
		fprintf(stderr, "CURAND error in file '%s', line %d, error: %s \nterminating!\n", __FILE__, __LINE__, \
			_curandReturnErrorString(err)); \
			assert(0); \
	}
}

extern "C" void curandCHECK(curandStatus_t err) { __curandCHECK(err, __FILE__, __LINE__); }

/*********************************************************/
/* FUNCTOR TO CHECK IF A POINT IS INSIDE THE UNIT CIRCLE */
/*********************************************************/
struct isInsideCircle {
    __device__ unsigned int operator() (float2 p) const {
        return (sqrtf(p.x * p.x + p.y * p.y) < 1.0f) ? 1 : 0;
    }
};

/********/
/* MAIN */
/********/
int main() {
	
	// --- Number of integration points
	const int N = 100000;

	thrust::device_vector<float2> d_p(N);

	curandGenerator_t rng;

	// --- Set the type of random number generator
	curandCHECK(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_MT19937));

	// --- Set the seed
	curandCHECK(curandSetPseudoRandomGeneratorSeed(rng, time(NULL)));

	// --- Generate N numbers in 2 dimensions
	curandCHECK(curandGenerateUniform(rng, (float *)thrust::raw_pointer_cast(&d_p[0]), 2 * N));

	// --- Count the points falling inside the unit circle
	unsigned int total = thrust::transform_reduce(d_p.begin(), d_p.end(), isInsideCircle(), 0, thrust::plus<unsigned int>());

	printf("The integral is %f\n", 4.0f * (float)total / (float)N);

	// --- Destroy the random number generator
	curandCHECK(curandDestroyGenerator(rng));

	return 0;

}

