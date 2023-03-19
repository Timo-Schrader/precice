#include <cstdlib>

#include <ginkgo/ginkgo.hpp>

#include "mapping/impl/BasisFunctions.hpp"

#define MAX_NUM_PREFETCH_ELEMENTS 1000

using precice::mapping::RadialBasisParameters;

template <typename ValueType, typename EvalFunctionType, unsigned int DefaultBlockSize>
__global__ void multiply_kernel_impl(std::size_t M, std::size_t N, ValueType *v1,  ValueType *v2,
                                    ValueType* x, ValueType *b, EvalFunctionType f, RadialBasisParameters params,  size_t v1RowLength,  size_t v2RowLength)
{

    __shared__ ValueType localB[DefaultBlockSize];
    __shared__ ValueType prefetchMemoryBuffer[MAX_NUM_PREFETCH_ELEMENTS + 3 * MAX_NUM_PREFETCH_ELEMENTS];// N + 3 * v2RowLength: N values of x, 3 * v2RowLength output vertex coordinates

    ValueType prefetchedEvalPoint[3]; // Unique for every row, hence not shared

    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto localIdx = i % DefaultBlockSize;

    ValueType dist = 0;
    ValueType y;

    // All blocks except for the last one can use barriers. The last block needs to disable some threads, hence no barrier is allowed here
    if(blockIdx.x < blockDim.x - 1){

        localB[localIdx] = 0;

        prefetchedEvalPoint[0] = v1[i];
        prefetchedEvalPoint[1] = v1[v1RowLength + i];
        prefetchedEvalPoint[2] = v1[2 * v1RowLength + i];

        for(size_t j = 0; j < v2RowLength; ++j){

            size_t localJ = j % MAX_NUM_PREFETCH_ELEMENTS;

            // Prefetch after every 1000 elements
            if(0 == localJ){
                if(0 == threadIdx.x){
                    // Check if amount of remaining elements is less than maximum number of prefetchable
                    for(size_t k = 0; k < min((uint)(v2RowLength - j), (uint)(MAX_NUM_PREFETCH_ELEMENTS)); ++k){
                        prefetchMemoryBuffer[k] = x[j + k];

                        prefetchMemoryBuffer[MAX_NUM_PREFETCH_ELEMENTS + 3 * k] = v2[j + k];
                        prefetchMemoryBuffer[MAX_NUM_PREFETCH_ELEMENTS + 3 * k + 1] = v2[v2RowLength + j + k];
                        prefetchMemoryBuffer[MAX_NUM_PREFETCH_ELEMENTS + 3 * k + 2] = v2[2 * v2RowLength + j + k];
                    }
                }
                __syncthreads();
            }

            dist = 0;
            for (size_t k = 0; k < 3; ++k) {
                y    = prefetchedEvalPoint[k] - prefetchMemoryBuffer[MAX_NUM_PREFETCH_ELEMENTS + 3 * localJ + k];
                dist = fma(y, y, dist);
            }

            localB[localIdx] +=  f(sqrt(dist), params) * prefetchMemoryBuffer[localJ];
        }

        b[i] = localB[localIdx];
    }
    else{
        if(i < M){
            localB[localIdx] = 0;

            prefetchedEvalPoint[0] = v1[i];
            prefetchedEvalPoint[1] = v1[v1RowLength + i];
            prefetchedEvalPoint[2] = v1[2 * v1RowLength + i];

            for(size_t j = 0; j < v2RowLength; ++j){
                dist = 0;
                for (size_t k = 0; k < 3; ++k) {
                    y    = prefetchedEvalPoint[k] - v2[k * v2RowLength + j];
                    dist = fma(y, y, dist);
                }

                localB[localIdx] +=  f(sqrt(dist), params) * x[j];
            }

            b[i] = localB[localIdx];
        }
    }
}

template <typename ValueType, typename EvalFunctionType>
void multiply_kernel(std::size_t M, std::size_t N,  ValueType *v1,  ValueType *v2,
    ValueType* x, ValueType *b, EvalFunctionType f, RadialBasisParameters params, std::size_t v1RowLength, std::size_t v2RowLength)
{
    constexpr unsigned int blockSize = 512;
    auto gridSize = (M + blockSize - 1) / blockSize;
    multiply_kernel_impl<ValueType, EvalFunctionType, blockSize><<<gridSize, blockSize>>>(M, N, v1, v2, x, b, f, params, v1RowLength, v2RowLength);
}


template void multiply_kernel<double, precice::mapping::CompactPolynomialC0>(std::size_t, std::size_t, double*, double*, double*, double*, precice::mapping::CompactPolynomialC0, RadialBasisParameters, std::size_t, std::size_t);
template void multiply_kernel<double, precice::mapping::CompactPolynomialC2>(std::size_t, std::size_t, double*, double*, double*, double*, precice::mapping::CompactPolynomialC2, RadialBasisParameters, std::size_t, std::size_t);
template void multiply_kernel<double, precice::mapping::CompactPolynomialC4>(std::size_t, std::size_t, double*, double*, double*, double*, precice::mapping::CompactPolynomialC4, RadialBasisParameters, std::size_t, std::size_t);
template void multiply_kernel<double, precice::mapping::CompactPolynomialC6>(std::size_t, std::size_t, double*, double*, double*, double*, precice::mapping::CompactPolynomialC6, RadialBasisParameters, std::size_t, std::size_t);
template void multiply_kernel<double, precice::mapping::CompactThinPlateSplinesC2>(std::size_t, std::size_t, double*, double*, double*, double*, precice::mapping::CompactThinPlateSplinesC2, RadialBasisParameters, std::size_t, std::size_t);
template void multiply_kernel<double, precice::mapping::Gaussian>(std::size_t, std::size_t, double*, double*, double*, double*, precice::mapping::Gaussian, RadialBasisParameters, std::size_t, std::size_t);
template void multiply_kernel<double, precice::mapping::VolumeSplines>(std::size_t, std::size_t, double*, double*, double*, double*, precice::mapping::VolumeSplines, RadialBasisParameters, std::size_t, std::size_t);
template void multiply_kernel<double, precice::mapping::ThinPlateSplines>(std::size_t, std::size_t, double*, double*, double*, double*, precice::mapping::ThinPlateSplines, RadialBasisParameters, std::size_t, std::size_t);
template void multiply_kernel<double, precice::mapping::Multiquadrics>(std::size_t, std::size_t, double*, double*, double*, double*, precice::mapping::Multiquadrics, RadialBasisParameters, std::size_t, std::size_t);
template void multiply_kernel<double, precice::mapping::InverseMultiquadrics>(std::size_t, std::size_t, double*, double*, double*, double*, precice::mapping::InverseMultiquadrics, RadialBasisParameters, std::size_t, std::size_t);
