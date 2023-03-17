#include <cstdlib>

#include <ginkgo/ginkgo.hpp>

#include "mapping/impl/BasisFunctions.hpp"

using precice::mapping::RadialBasisParameters;

template <typename ValueType, typename EvalFunctionType, unsigned int DefaultBlockSize>
__global__ void multiply_kernel_impl(std::size_t M, std::size_t N, ValueType *v1,  ValueType *v2,
                                    ValueType* x, ValueType *b, EvalFunctionType f, RadialBasisParameters params,  size_t v1RowLength,  size_t v2RowLength)
{

    __shared__ ValueType localB[DefaultBlockSize];

    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto localIdx = i % DefaultBlockSize;

    if(i < M){
        ValueType dist = 0;
        ValueType y;

        localB[localIdx] = 0;

        ValueType prefetchedEvalPoint[3];

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
