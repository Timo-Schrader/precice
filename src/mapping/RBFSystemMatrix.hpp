#pragma once

#include <iostream>
#include <map>
#include <string>

#include <ginkgo/ginkgo.hpp>
#include <omp.h>

#include "mapping/impl/BasisFunctions.hpp"

using GinkgoVector = gko::matrix::Dense<>;
using precice::mapping::RadialBasisParameters;

// A CUDA kernel implementing the stencil, which will be used if running on the
// CUDA executor. Unfortunately, NVCC has serious problems interpreting some
// parts of Ginkgo's code, so the kernel has to be compiled separately.
template <typename ValueType, typename EvalFunctionType>
void multiply_kernel(std::size_t size, ValueType *v1, ValueType *v2,
                     ValueType *x, ValueType *b, EvalFunctionType f, RadialBasisParameters params, size_t v1RowLength, size_t v2RowLength);

template <typename ValueType, typename EvalFunctionType>
class RBFSystemMatrix : public gko::EnableLinOp<RBFSystemMatrix<ValueType, EvalFunctionType>>,
                        public gko::EnableCreateMethod<RBFSystemMatrix<ValueType, EvalFunctionType>> {
public:
  // This ructor will be called by the create method. Here we initialize
  // the coefficients of the stencil.
  RBFSystemMatrix(std::shared_ptr<const gko::Executor> exec, std::size_t N = 0, std::shared_ptr<GinkgoVector> inputVertices = std::shared_ptr<GinkgoVector>(), std::shared_ptr<GinkgoVector> outputVertices = std::shared_ptr<GinkgoVector>(), EvalFunctionType *f = nullptr)
      : gko::EnableLinOp<RBFSystemMatrix>(exec, gko::dim<2>{N}), _N(N), _f(*f)
  {
    _inputVertices  = inputVertices;
    _outputVertices = outputVertices;
  }

protected:
  std::shared_ptr<GinkgoVector> _inputVertices;
  std::shared_ptr<GinkgoVector> _outputVertices;
  std::size_t                   _N;
  mutable EvalFunctionType      _f;

  // Here we implement the application of the linear operator, x = A * b.
  // apply_impl will be called by the apply method, after the arguments have
  // been moved to the correct executor and the operators checked for
  // conforming sizes.
  //
  // For simplicity, we assume that there is always only one right hand side
  // and the stride of consecutive elements in the vectors is 1 (both of these
  // are always true in this example).
  void apply_impl(const gko::LinOp *x, gko::LinOp *b) const override
  {
    // we only implement the operator for dense RHS.
    // gko::as will throw an exception if its argument is not Dense.
    auto _x = gko::as<GinkgoVector>(const_cast<gko::LinOp *>(x));
    auto _b = gko::as<GinkgoVector>(b);

    // we need separate implementations depending on the executor, so we
    // create an operation which maps the call to the correct implementation
    struct stencil_operation : gko::Operation {
      stencil_operation(std::size_t N, GinkgoVector *v1, GinkgoVector *v2,
                        GinkgoVector *x, GinkgoVector *b, EvalFunctionType f, RadialBasisParameters params, std::size_t v1RowLength, std::size_t v2RowLength)
          : _N{N}, _f{f}, _params(params), _v1RowLength{v1RowLength}, _v2RowLength{v2RowLength}
      {
        _x  = x->get_values();
        _b  = b->get_values();
        _v1 = v1->get_values();
        _v2 = v2->get_values();
      }

      // CUDA implementation
      void run(std::shared_ptr<const gko::CudaExecutor>) const override
      {
        multiply_kernel(_N, _v1, _v2, _x, _b, _f, _params, _v1RowLength, _v2RowLength);
      }

      // We do not provide an implementation for reference executor.
      // If not provided, Ginkgo will use the implementation for the
      // OpenMP executor when calling it in the reference executor.

      std::size_t           _N;
      ValueType *           _v1;
      ValueType *           _v2;
      ValueType *           _b;
      ValueType *           _x; // Shallow copy!
      EvalFunctionType      _f;
      RadialBasisParameters _params;
      std::size_t           _v1RowLength;
      std::size_t           _v2RowLength;
    };
    this->get_executor()->run(
        stencil_operation(_N, gko::lend(_inputVertices), gko::lend(_inputVertices), gko::lend(_x), gko::lend(_b), _f, _f.getFunctionParameters(), _inputVertices->get_size()[1], _inputVertices->get_size()[1]));
  }

  void apply_impl(const gko::LinOp *alpha, const gko::LinOp *b,
                  const gko::LinOp *beta, gko::LinOp *x) const override
  {
  }
};
