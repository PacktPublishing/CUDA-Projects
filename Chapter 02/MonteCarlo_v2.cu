#include <curand.h>
#include <curand_kernel.h>

#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <time.h>

/************************************/
/* RANDOM NUMBERS GENERATOR FUNCTOR */
/************************************/
struct estimationHelper : public thrust::unary_function<unsigned int, float>
{
	unsigned int m_seed;

	__host__ __device__ estimationHelper(unsigned int seed) { m_seed = seed; }

	__device__ float operator()(const unsigned int n) {

        curandState s;

        // --- Seed a random number generator
        curand_init(m_seed, n, 0, &s);

       	// --- Generate random point in the unit square
        float x = curand_uniform(&s);
        float y = curand_uniform(&s);

        return (sqrtf(x * x + y * y) < 1.0f) ? 1 : 0;

    }
};

/********/
/* MAIN */
/********/
int main() {

	// --- Number of integration points
	const int N = 100000;

	unsigned int total = thrust::transform_reduce(thrust::counting_iterator<unsigned int>(0), thrust::counting_iterator<unsigned int>(N),
	                                          estimationHelper(time(NULL)), 0, thrust::plus<unsigned int>());

	printf("The integral is %f\n", 4.f * (float)total / (float)N);

	return 0;
}

