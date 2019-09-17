#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/transform_reduce.h>
#include <time.h>

/***********************************/
/* RANDOM POINT GENERATION FUNCTOR */
/***********************************/
struct generateRandomPoint{

	unsigned int m_seed;

	__host__ __device__ generateRandomPoint(unsigned int seed) { m_seed = seed; }

	__device__ float2 operator() (const unsigned int n) {

    	thrust::default_random_engine rng(m_seed);

    	rng.discard(2 * n);

    	thrust::uniform_real_distribution<float> u(-1.f, 1.f);

    	return make_float2(u(rng), u(rng));
    }
};

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

	// --- Generate random points within a unit square
	thrust::device_vector<float2> d_p(N);
	thrust::transform(thrust::counting_iterator<unsigned int>(0), thrust::counting_iterator<unsigned int>(N), d_p.begin(), generateRandomPoint(time(0)));

	// --- Count the points falling inside the unit circle
	unsigned int total = thrust::transform_reduce(d_p.begin(), d_p.end(), isInsideCircle(), 0, thrust::plus<unsigned int>());

	printf("The integral is %f\n", 4.0f * (float)total / (float)N);
	
	return 0;
}

