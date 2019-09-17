#include <moderngpu/kernel_reduce.hxx>
#include <moderngpu/memory.hxx>

#define BLOCKSIZE 256
#define GRAINSIZE 4

#define pi_f  3.14159265358979f                 // Greek pi in single precision

/********/
/* MAIN */
/********/
int main() {

	// --- Integration domain
	float a = 0.5f;
	float b = 1.f;

	// --- Number of integration nodes
	const int N = 1024 * 256;

	// --- Generate sampling points
	float h = (b - a) / (float)(N - 1);  // --- The number of discretization intervals is the number of integration nodes minus one

	// --- context_t is an abstract base class through which CUDA runtime features like cudaMalloc and cudaFree
	//     may be accessed. standard_context_t is its implementation.
	mgpu::standard_context_t context;

    mgpu::mem_t<float> d_s(1, context);

    // --- Lambda function implementing the weights x function sample products
    auto f = [=]MGPU_DEVICE(int tidx) {

    	float coeff;

    	if (tidx & 1 == 1) coeff = 2.f;
    	else coeff = 4.f;

    	if ((tidx == 0) || (tidx == N - 1)) coeff = 1.f;

    	float x = a + tidx * h;

    	// --- Quadrature weights x function samples evaluation
    	return (h / 3.f) * coeff * sin(2.f * pi_f * x); };

    // --- Transform-reduce to numerically calculate the integral
    transform_reduce(f, N, d_s.data(), mgpu::plus_t<float>(), context);

    // --- Move the result from GPU to CPU
    std::vector<float> h_s = from_mem(d_s);

    printf("The integral is %f \n", h_s[0]);

    return 0;

}
