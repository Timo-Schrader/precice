#pragma once

#include <iostream>
#include <map>
#include <string>

#include <ginkgo/ginkgo.hpp>
#include <omp.h>

#include "mapping/impl/BasisFunctions.hpp"

using GinkgoVector = gko::matrix::Dense<>;
using precice::mapping::RadialBasisParameters;

template <typename ValueType, typename EvalFunctionType>
void multiply_kernel(std::size_t M, std::size_t N, ValueType *v1, ValueType *v2,
                     ValueType *x, ValueType *b, EvalFunctionType f, RadialBasisParameters params, size_t v1RowLength, size_t v2RowLength);

template <typename ValueType, typename EvalFunctionType>
class RBFMatrix : public gko::EnableLinOp<RBFMatrix<ValueType, EvalFunctionType>>,
                  public gko::EnableCreateMethod<RBFMatrix<ValueType, EvalFunctionType>> {
public:
  RBFMatrix(std::shared_ptr<const gko::Executor> exec, std::size_t M = 0, std::size_t N = 0, std::shared_ptr<GinkgoVector> inputVertices = std::shared_ptr<GinkgoVector>(), std::shared_ptr<GinkgoVector> outputVertices = std::shared_ptr<GinkgoVector>(), EvalFunctionType *f = nullptr)
      : gko::EnableLinOp<RBFMatrix>(exec, gko::dim<2>{M, N}), _M(M), _N(N), _f(*f)
  {
    _inputVertices  = inputVertices;
    _outputVertices = outputVertices;
  }

protected:
  std::shared_ptr<GinkgoVector> _inputVertices;
  std::shared_ptr<GinkgoVector> _outputVertices;
  std::size_t                   _M;
  std::size_t                   _N;
  mutable EvalFunctionType      _f;

  void apply_impl(const gko::LinOp *x, gko::LinOp *b) const override
  {
    auto _x = gko::as<GinkgoVector>(const_cast<gko::LinOp *>(x));
    auto _b = gko::as<GinkgoVector>(b);

    // we need separate implementations depending on the executor, so we
    // create an operation which maps the call to the correct implementation

    this->get_executor()->run(
        stencil_operation(_M, _N, gko::lend(_inputVertices), gko::lend(_outputVertices), gko::lend(_x), gko::lend(_b), _f, _f.getFunctionParameters(), _inputVertices->get_size()[1], _outputVertices->get_size()[1]));
  }

  void apply_impl(const gko::LinOp *alpha, const gko::LinOp *x,
                  const gko::LinOp *beta, gko::LinOp *b) const override
  {
    // auto _x = gko::as<GinkgoVector>(const_cast<gko::LinOp *>(x));
    // auto _b = gko::as<GinkgoVector>(b);
    //
    // this->get_executor()->run(
    //    stencil_operation(_N, gko::lend(_inputVertices), gko::lend(_outputVertices), gko::lend(_x), gko::lend(_b), _f, _f.getFunctionParameters(), _inputVertices->get_size()[1], _outputVertices->get_size()[1]));
  }

  struct stencil_operation : gko::Operation {
    stencil_operation(std::size_t M, std::size_t N, GinkgoVector *v1, GinkgoVector *v2,
                      GinkgoVector *x, GinkgoVector *b, EvalFunctionType f, RadialBasisParameters params, std::size_t v1RowLength, std::size_t v2RowLength)
        : _M(M), _N{N}, _f{f}, _params(params), _v1RowLength{v1RowLength}, _v2RowLength{v2RowLength}
    {
      _x  = x->get_values();
      _b  = b->get_values();
      _v1 = v1->get_values();
      _v2 = v2->get_values();
    }

    // CUDA implementation
    void run(std::shared_ptr<const gko::CudaExecutor>) const override
    {
      multiply_kernel(_M, _N, _v1, _v2, _x, _b, _f, _params, _v1RowLength, _v2RowLength);
    }

    std::size_t           _M;
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
};
