#ifndef RTORCH_H
#define RTORCH_H

#define R_NO_REMAP

// Step 1: RcppCommon.h (declares Rcpp namespace, traits, but not full Rcpp)
#include <RcppCommon.h>

// Step 2: libtorch headers
#include <torch/torch.h>

// Step 3: Forward-declare Rcpp::as/wrap specializations (before Rcpp.h)
namespace Rcpp {
    template<> at::Tensor as(SEXP);
    template<> SEXP wrap(const at::Tensor&);
}

// Step 4: Full Rcpp (sees the forward declarations above)
#include <Rcpp.h>

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

// Convert R SEXP to optional tensor (NULL → nullopt)
c10::optional<at::Tensor> sexp_to_optional_tensor(SEXP x);

// Convert R list of tensors to std::vector<at::Tensor>
std::vector<at::Tensor> sexp_to_tensor_list(SEXP x);

// Convert R SEXP to optional primitives (NULL → nullopt)
c10::optional<int64_t> sexp_to_optional_int(SEXP x);
c10::optional<double> sexp_to_optional_double(SEXP x);
c10::optional<bool> sexp_to_optional_bool(SEXP x);

// Convert R SEXP to optional string (NULL → nullopt)
c10::optional<std::string> sexp_to_optional_string(SEXP x);

// Convert R SEXP to optional scalar (NULL → nullopt)
c10::optional<at::Scalar> sexp_to_optional_scalar(SEXP x);

// Convert R SEXP to optional device (NULL → nullopt)
c10::optional<at::Device> sexp_to_optional_device(SEXP x);

// Convert R SEXP to non-optional device
at::Device sexp_to_required_device(SEXP x);

// Convert R SEXP to optional generator (always nullopt for now)
c10::optional<at::Generator> sexp_to_optional_generator(SEXP x);

// Convert R SEXP to optional memory format
c10::optional<at::MemoryFormat> sexp_to_optional_memory_format(SEXP x);

// Convert R numeric vector to std::vector<double>
std::vector<double> sexp_to_double_vec(SEXP x);
c10::optional<std::vector<double>> sexp_to_optional_double_vec(SEXP x);

// Convert R numeric/integer vector to std::vector<at::Scalar>
std::vector<at::Scalar> sexp_to_scalar_list(SEXP x);

// Convert R list of tensors (with possible NULLs) to c10::List of optional tensors
c10::List<c10::optional<at::Tensor>> sexp_to_optional_tensor_list(SEXP x);

// Convert std::vector<at::Tensor> to R list of tensors
SEXP tensor_list_to_sexp(const std::vector<at::Tensor>& tensors);

#endif // RTORCH_H
