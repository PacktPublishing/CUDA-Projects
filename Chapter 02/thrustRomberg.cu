#include <thrust/sequence.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define pi_f  3.14159265358979f                 // Greek pi in single precision

/*********************/
/* INTEGRAND FUNCTOR */
/*********************/
struct sin_functor { __host__ __device__ float operator()(float x) const { return sin(2.f * pi_f * x); } };

/********/
/* MAIN */
/********/
int main() {
	
	// --- Integration domain
	float a = 0.5f;
	float b = 1.f;

	// --- Maximum number of Romberg iterations
	int Kmax = 5;								

	// --- Define the matrix for Romberg approximations and initialize to 1.f 
	thrust::host_vector<float> R(Kmax * Kmax, 1.f);

	// --- Compute the first column of Romberg matrix
	for (int k = 0; k < Kmax; k++) {

		// --- Step size for the k-th row of the Romberg matrix
		float h = (b - a) / pow(2.f, k + 1);

		// --- Define integration nodes
		int N = (int)((b - a) / h) + 1;
		thrust::device_vector<float> d_x(N);
		thrust::sequence(d_x.begin(), d_x.end(), a, h);

		// --- Calculate function values
		thrust::device_vector<float> d_y(N);
		thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), sin_functor());

		// --- Calculate integral
		R[k * Kmax] = (.5f * h) * (d_y[0] + 2.f * thrust::reduce(d_y.begin() + 1, d_y.begin() + N - 1, 0.0f) + d_y[N - 1]);
	}

	// --- Compute the other columns of the Romberg matrix. Remember that the Romberg matrix is triangular.
	for (int k = 1; k < Kmax; k++) {
		
		for (int j = 1; j <= k; j++) {

			// --- Computing R[k, j]
			R[k * Kmax + j] = R[k * Kmax + j - 1] + (R[k * Kmax + j - 1] - R[(k - 1) * Kmax + j - 1]) / (pow(4.f, j) - 1.f);

		}
	}

	// --- Define the vector Rnum for numerical approximations
	printf("The integral is %f\n", R[Kmax * Kmax - 1]);

	return 0;
}
