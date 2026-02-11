#ifndef RTORCH_H
#define RTORCH_H

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

#include <torch/torch.h>

// ---- Tensor helpers ----

// Wrap a heap-allocated at::Tensor into an R external pointer with class
SEXP make_tensor_sexp(at::Tensor* t);

// Extract at::Tensor* from R external pointer (errors if null)
at::Tensor* get_tensor_ptr(SEXP x);

// ---- Scalar conversion helpers ----

// Convert R SEXP to at::Scalar (numeric or integer)
at::Scalar sexp_to_scalar(SEXP x);

// Convert R SEXP to c10::optional<at::ScalarType>
c10::optional<at::ScalarType> sexp_to_dtype(SEXP dtype);

// Convert R integer/numeric vector to IntArrayRef (caller owns storage)
std::vector<int64_t> sexp_to_int_vec(SEXP x);

// Convert R string SEXP to at::Device (NULL → CPU)
at::Device sexp_to_device(SEXP device_sexp);

#endif // RTORCH_H
