#include <thrust/inner_product.h>
#include <thrust/device_vector.h>

#define STRIDE 2

#define pi_f  3.14159265358979f        // Greek pi in single precision

/*********************/
/* INTEGRAND FUNCTOR */
/*********************/
struct sin_functor { __host__ __device__ float operator()(float x) const { return sin(2.f * pi_f * x); } };

/***********************/
/* STEPPED RANGE CLASS */
/***********************/
template <typename templateIterator>
class steppedRangeClass {

public:

	typedef typename thrust::iterator_difference<templateIterator>::type differenceIteratorType;

	struct relevantFunctor : public thrust::unary_function<differenceIteratorType, differenceIteratorType>
	{
		differenceIteratorType step;

		relevantFunctor(differenceIteratorType step) : step(step) {}

		__host__ __device__ differenceIteratorType operator()(const differenceIteratorType& k) const { return step * k; }
	};

	typedef typename thrust::counting_iterator<differenceIteratorType> countingIteratorDifferenceType;

	typedef typename thrust::transform_iterator<relevantFunctor, countingIteratorDifferenceType> transformIteratorStep;

	typedef typename thrust::permutation_iterator<templateIterator, transformIteratorStep> permutationIteratorStep;

	// --- Type of the stepped range iterator
	typedef permutationIteratorStep it;

	// --- Construct stepped range for the range [first, last)
	steppedRangeClass(templateIterator firstElement, templateIterator lastElement, differenceIteratorType step) : firstElement(firstElement), lastElement(lastElement), step(step) {}

	it begin(void) const { return permutationIteratorStep(firstElement, transformIteratorStep(countingIteratorDifferenceType(0), relevantFunctor(step))); }

	it end(void) const { return begin() + ((lastElement - firstElement) + (step - 1)) / step; }

protected:

	templateIterator firstElement;
	templateIterator lastElement;
	differenceIteratorType step;
};

/********/
/* MAIN */
/********/
int main() {

	// --- Integration domain
	float a = 0.5f;
	float b = 1.f;

	// --- Number of integration nodes
	const int N = 1000;
	
	// --- Generate the two integration coefficients
	thrust::host_vector<float> h_coefficients(STRIDE);
	h_coefficients[0] = 4.f;
	h_coefficients[1] = 2.f;

	// --- Construct the vector of integration weights
	thrust::device_vector<float> d_coefficients(N);
	
	typedef thrust::device_vector<float>::iterator templateIterator;
	
	steppedRangeClass<templateIterator> pos1(d_coefficients.begin() + 1, d_coefficients.end() - 2, STRIDE);
	steppedRangeClass<templateIterator> pos2(d_coefficients.begin() + 2, d_coefficients.end() - 1, STRIDE);

	thrust::fill(pos1.begin(), pos1.end(), h_coefficients[0]);
	thrust::fill(pos2.begin(), pos2.end(), h_coefficients[1]);

	// --- Setting the first and last coefficients separately
	d_coefficients[0]		= 1.f;
	d_coefficients[N - 1]	= 1.f;

	// --- Print out the generated d_coefficients
	//std::cout << "d_coefficients: ";
	//thrust::copy(d_coefficients.begin(), d_coefficients.end(), std::ostream_iterator<float>(std::cout, " "));  std::cout << std::endl;

	// --- Generate sampling points
	float h = (b - a) / (float)(N - 1);  // --- The number of discretization intervals is the number of integration nodes minus one

	thrust::device_vector<float> d_x(N);
	thrust::transform(thrust::make_counting_iterator(a / h),
					  thrust::make_counting_iterator((b + 1.f) / h),
					  thrust::make_constant_iterator(h),
				      d_x.begin(),
					  thrust::multiplies<float>());

	// --- Print out the generated sampling points d_x
	// std::cout << "d_x: ";
	// thrust::copy(d_x.begin(), d_x.end(), std::ostream_iterator<float>(std::cout, " "));  std::cout << std::endl;

	// --- Calculate function values
	thrust::device_vector<float> d_y(N);
	thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), sin_functor());

	// --- Print out the generated samples d_x
	// std::cout << "d_y: ";
	// thrust::copy(d_y.begin(), d_y.end(), std::ostream_iterator<float>(std::cout, " "));  std::cout << std::endl;

	// --- Calculate integral
	float integral = (h / 3.f) * thrust::inner_product(d_y.begin(), d_y.begin() + N, d_coefficients.begin(), 0.0f);
	printf("The integral is %f\n", integral);

	return 0;

}
