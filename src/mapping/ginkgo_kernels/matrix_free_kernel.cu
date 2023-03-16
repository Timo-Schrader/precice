/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <cstdlib>

#include <ginkgo/ginkgo.hpp>

#include "mapping/impl/BasisFunctions.hpp"

using precice::mapping::RadialBasisParameters;

namespace {


// a parallel CUDA kernel that computes the application of a 3 point stencil
template <typename ValueType, typename EvalFunctionType>
__global__ void multiply_kernel_impl(std::size_t N,  ValueType *v1,  ValueType *v2,
                                    ValueType* x, ValueType *b, EvalFunctionType f, RadialBasisParameters params,  size_t v1RowLength,  size_t v2RowLength)
{

    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N){
        double dist = 0;
        double y;

        b[i] = 0;

        double prefetchedEvalPoint[3];

        prefetchedEvalPoint[0] = v1[0 * v1RowLength + i];
        prefetchedEvalPoint[1] = v1[1 * v1RowLength + i];
        prefetchedEvalPoint[2] = v1[2 * v1RowLength + i];

        for(size_t j = 0; j < v2RowLength; ++j){
            dist = 0;
            for (size_t k = 0; k < 3; ++k) {
                y    = prefetchedEvalPoint[k] - v2[k * v2RowLength + j];
                dist = fma(y, y, dist);
            }

            b[i] +=  f(sqrt(dist), params) * x[j];
        }
    }
}


}  // namespace


template <typename ValueType, typename EvalFunctionType>
void multiply_kernel(std::size_t size,  ValueType *v1,  ValueType *v2,
    ValueType* x, ValueType *b, EvalFunctionType f, RadialBasisParameters params, std::size_t v1RowLength, std::size_t v2RowLength)
{
    constexpr int block_size = 512;
    auto grid_size = (size + block_size - 1) / block_size;
    multiply_kernel_impl<<<grid_size, block_size>>>(size, v1, v2, x, b, f, params, v1RowLength, v2RowLength);
}


template void multiply_kernel<double, precice::mapping::CompactPolynomialC0>(size_t, double*, double*, double*, double*, precice::mapping::CompactPolynomialC0, RadialBasisParameters, std::size_t, std::size_t);
template void multiply_kernel<double, precice::mapping::CompactPolynomialC2>(size_t, double*, double*, double*, double*, precice::mapping::CompactPolynomialC2, RadialBasisParameters, std::size_t, std::size_t);
template void multiply_kernel<double, precice::mapping::CompactPolynomialC4>(size_t, double*, double*, double*, double*, precice::mapping::CompactPolynomialC4, RadialBasisParameters, std::size_t, std::size_t);
template void multiply_kernel<double, precice::mapping::CompactPolynomialC6>(size_t, double*, double*, double*, double*, precice::mapping::CompactPolynomialC6, RadialBasisParameters, std::size_t, std::size_t);
template void multiply_kernel<double, precice::mapping::CompactThinPlateSplinesC2>(size_t, double*, double*, double*, double*, precice::mapping::CompactThinPlateSplinesC2, RadialBasisParameters, std::size_t, std::size_t);
template void multiply_kernel<double, precice::mapping::Gaussian>(size_t, double*, double*, double*, double*, precice::mapping::Gaussian, RadialBasisParameters, std::size_t, std::size_t);
template void multiply_kernel<double, precice::mapping::VolumeSplines>(size_t, double*, double*, double*, double*, precice::mapping::VolumeSplines, RadialBasisParameters, std::size_t, std::size_t);
template void multiply_kernel<double, precice::mapping::ThinPlateSplines>(size_t, double*, double*, double*, double*, precice::mapping::ThinPlateSplines, RadialBasisParameters, std::size_t, std::size_t);
template void multiply_kernel<double, precice::mapping::Multiquadrics>(size_t, double*, double*, double*, double*, precice::mapping::Multiquadrics, RadialBasisParameters, std::size_t, std::size_t);
template void multiply_kernel<double, precice::mapping::InverseMultiquadrics>(size_t, double*, double*, double*, double*, precice::mapping::InverseMultiquadrics, RadialBasisParameters, std::size_t, std::size_t);
