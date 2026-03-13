// Fused SIMD kernels for tinytorch (originally from torchlang)
// Uses ATen's at::vec API for vectorization

#include "tinytorch.h"
#include <ATen/cpu/vec/vec.h>
#include <cmath>
#include <algorithm>

// Use tinytorch's get_tensor_ptr and make_tensor_sexp from tinytorch.h
// Wrapper to create tensor from rvalue ref using tinytorch's heap-alloc convention
static SEXP make_fused_tensor(at::Tensor&& t) {
  return make_tensor_sexp(new at::Tensor(std::move(t)));
}

// ============================================================================
// Fused kernels
// ============================================================================

// [[Rcpp::export]]
SEXP cpp_fused_relu(SEXP input_sexp) {
  at::Tensor* input = get_tensor_ptr(input_sexp);
  if (!input) {
    Rf_error("Invalid torch tensor input");
    return R_NilValue;
  }

  at::Tensor output = at::empty_like(*input);
  int64_t n = input->numel();

  float* px = input->data_ptr<float>();
  float* pout = output.data_ptr<float>();

  using Vec = at::vec::Vectorized<float>;
  constexpr int64_t vec_size = Vec::size();
  int64_t vec_end = (n / vec_size) * vec_size;

  #pragma omp parallel for
  for (int64_t i = 0; i < vec_end; i += vec_size) {
    Vec vx = Vec::loadu(px + i);
    Vec v1 = at::vec::clamp_min(vx, Vec(0.0f));
    v1.store(pout + i);
  }

  for (int64_t i = vec_end; i < n; i++) {
    pout[i] = std::max(px[i], 0.0f);
  }

  return make_fused_tensor(std::move(output));
}

// [[Rcpp::export]]
SEXP cpp_fused_relu_sigmoid(SEXP input_sexp) {
  at::Tensor* input = get_tensor_ptr(input_sexp);
  if (!input) {
    Rf_error("Invalid torch tensor input");
    return R_NilValue;
  }

  at::Tensor output = at::empty_like(*input);
  int64_t n = input->numel();

  float* px = input->data_ptr<float>();
  float* pout = output.data_ptr<float>();

  using Vec = at::vec::Vectorized<float>;
  constexpr int64_t vec_size = Vec::size();
  int64_t vec_end = (n / vec_size) * vec_size;

  #pragma omp parallel for
  for (int64_t i = 0; i < vec_end; i += vec_size) {
    Vec vx = Vec::loadu(px + i);
    Vec v1 = at::vec::clamp_min(vx, Vec(0.0f));
    Vec v2 = Vec(1.0f) / (Vec(1.0f) + (Vec(0.0f) - v1).exp());
    v2.store(pout + i);
  }

  for (int64_t i = vec_end; i < n; i++) {
    float s1 = std::max(px[i], 0.0f);
    pout[i] = 1.0f / (1.0f + std::exp(-s1));
  }

  return make_fused_tensor(std::move(output));
}

// [[Rcpp::export]]
SEXP cpp_fused_relu_sigmoid_tanh(SEXP input_sexp) {
  at::Tensor* input = get_tensor_ptr(input_sexp);
  if (!input) {
    Rf_error("Invalid torch tensor input");
    return R_NilValue;
  }

  at::Tensor output = at::empty_like(*input);
  int64_t n = input->numel();

  float* px = input->data_ptr<float>();
  float* pout = output.data_ptr<float>();

  using Vec = at::vec::Vectorized<float>;
  constexpr int64_t vec_size = Vec::size();
  int64_t vec_end = (n / vec_size) * vec_size;

  #pragma omp parallel for
  for (int64_t i = 0; i < vec_end; i += vec_size) {
    Vec vx = Vec::loadu(px + i);
    Vec v1 = at::vec::clamp_min(vx, Vec(0.0f));
    Vec v2 = Vec(1.0f) / (Vec(1.0f) + (Vec(0.0f) - v1).exp());
    Vec v3 = v2.tanh();
    v3.store(pout + i);
  }

  for (int64_t i = vec_end; i < n; i++) {
    float s1 = std::max(px[i], 0.0f);
    float s2 = 1.0f / (1.0f + std::exp(-s1));
    pout[i] = std::tanh(s2);
  }

  return make_fused_tensor(std::move(output));
}

// ============================================================================
// Tier 1: Common activation patterns
// ============================================================================

// Fused SiLU (Swish): x * sigmoid(x)
// Used in LLaMA, Mistral, modern LLMs
// [[Rcpp::export]]
SEXP cpp_fused_silu(SEXP input_sexp) {
  at::Tensor* input = get_tensor_ptr(input_sexp);
  if (!input) {
    Rf_error("Invalid torch tensor input");
    return R_NilValue;
  }

  at::Tensor output = at::empty_like(*input);
  int64_t n = input->numel();

  float* px = input->data_ptr<float>();
  float* pout = output.data_ptr<float>();

  using Vec = at::vec::Vectorized<float>;
  constexpr int64_t vec_size = Vec::size();
  int64_t vec_end = (n / vec_size) * vec_size;

  #pragma omp parallel for
  for (int64_t i = 0; i < vec_end; i += vec_size) {
    Vec vx = Vec::loadu(px + i);
    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    Vec sigmoid = Vec(1.0f) / (Vec(1.0f) + (Vec(0.0f) - vx).exp());
    Vec result = vx * sigmoid;
    result.store(pout + i);
  }

  for (int64_t i = vec_end; i < n; i++) {
    float x = px[i];
    pout[i] = x / (1.0f + std::exp(-x));
  }

  return make_fused_tensor(std::move(output));
}

// Fused GELU (Gaussian Error Linear Unit)
// Approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Used in GPT, BERT, transformers
// [[Rcpp::export]]
SEXP cpp_fused_gelu(SEXP input_sexp) {
  at::Tensor* input = get_tensor_ptr(input_sexp);
  if (!input) {
    Rf_error("Invalid torch tensor input");
    return R_NilValue;
  }

  at::Tensor output = at::empty_like(*input);
  int64_t n = input->numel();

  float* px = input->data_ptr<float>();
  float* pout = output.data_ptr<float>();

  using Vec = at::vec::Vectorized<float>;
  constexpr int64_t vec_size = Vec::size();
  int64_t vec_end = (n / vec_size) * vec_size;

  // Constants for GELU approximation
  const float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/pi)
  const float coeff = 0.044715f;

  #pragma omp parallel for
  for (int64_t i = 0; i < vec_end; i += vec_size) {
    Vec vx = Vec::loadu(px + i);
    // gelu(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Vec x3 = vx * vx * vx;
    Vec inner = Vec(sqrt_2_over_pi) * (vx + Vec(coeff) * x3);
    Vec result = vx * Vec(0.5f) * (Vec(1.0f) + inner.tanh());
    result.store(pout + i);
  }

  for (int64_t i = vec_end; i < n; i++) {
    float x = px[i];
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    pout[i] = x * 0.5f * (1.0f + std::tanh(inner));
  }

  return make_fused_tensor(std::move(output));
}

// Fused Sin+Cos: compute both sin and cos in one pass
// Used in positional encodings (RoPE, sinusoidal)
// Returns an R list with two tensors: list(sin=..., cos=...)
// [[Rcpp::export]]
SEXP cpp_fused_sincos(SEXP input_sexp) {
  at::Tensor* input = get_tensor_ptr(input_sexp);
  if (!input) {
    Rf_error("Invalid torch tensor input");
    return R_NilValue;
  }

  at::Tensor sin_result = at::empty_like(*input);
  at::Tensor cos_result = at::empty_like(*input);
  int64_t n = input->numel();

  float* px = input->data_ptr<float>();
  float* psin = sin_result.data_ptr<float>();
  float* pcos = cos_result.data_ptr<float>();

  using Vec = at::vec::Vectorized<float>;
  constexpr int64_t vec_size = Vec::size();
  int64_t vec_end = (n / vec_size) * vec_size;

  #pragma omp parallel for
  for (int64_t i = 0; i < vec_end; i += vec_size) {
    Vec vx = Vec::loadu(px + i);
    Vec vsin = vx.sin();
    Vec vcos = vx.cos();
    vsin.store(psin + i);
    vcos.store(pcos + i);
  }

  for (int64_t i = vec_end; i < n; i++) {
    psin[i] = std::sin(px[i]);
    pcos[i] = std::cos(px[i]);
  }

  // Create R list with named elements
  SEXP result = PROTECT(Rf_allocVector(VECSXP, 2));
  SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));

  SET_STRING_ELT(names, 0, Rf_mkChar("sin"));
  SET_STRING_ELT(names, 1, Rf_mkChar("cos"));
  Rf_setAttrib(result, R_NamesSymbol, names);

  SET_VECTOR_ELT(result, 0, make_fused_tensor(std::move(sin_result)));
  SET_VECTOR_ELT(result, 1, make_fused_tensor(std::move(cos_result)));

  UNPROTECT(2);
  return result;
}

// Fused softcap: (x / cap).tanh() * cap
// Used in Gemma3 attention for logit stabilization
// [[Rcpp::export]]
SEXP cpp_fused_softcap(SEXP input_sexp, SEXP cap_sexp) {
  at::Tensor* input = get_tensor_ptr(input_sexp);
  if (!input) {
    Rf_error("Invalid torch tensor input");
    return R_NilValue;
  }

  float cap = (float)Rf_asReal(cap_sexp);

  at::Tensor output = at::empty_like(*input);
  int64_t n = input->numel();

  float* px = input->data_ptr<float>();
  float* pout = output.data_ptr<float>();

  using Vec = at::vec::Vectorized<float>;
  constexpr int64_t vec_size = Vec::size();
  int64_t vec_end = (n / vec_size) * vec_size;

  Vec vcap(cap);

  #pragma omp parallel for
  for (int64_t i = 0; i < vec_end; i += vec_size) {
    Vec vx = Vec::loadu(px + i);
    // softcap(x) = tanh(x / cap) * cap
    Vec result = (vx / vcap).tanh() * vcap;
    result.store(pout + i);
  }

  for (int64_t i = vec_end; i < n; i++) {
    pout[i] = std::tanh(px[i] / cap) * cap;
  }

  return make_fused_tensor(std::move(output));
}

// Fused RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
// Note: This operates on the LAST dimension only
// Used in LLaMA, modern transformers (replaces LayerNorm)
// [[Rcpp::export]]
SEXP cpp_fused_rmsnorm(SEXP input_sexp, SEXP weight_sexp, SEXP eps_sexp) {
  at::Tensor* input = get_tensor_ptr(input_sexp);
  at::Tensor* weight = get_tensor_ptr(weight_sexp);
  if (!input || !weight) {
    Rf_error("Invalid torch tensor input");
    return R_NilValue;
  }

  float eps = (float)Rf_asReal(eps_sexp);

  // Get dimensions
  auto sizes = input->sizes();
  int64_t last_dim = sizes[sizes.size() - 1];
  int64_t outer_size = input->numel() / last_dim;

  at::Tensor output = at::empty_like(*input);

  float* px = input->data_ptr<float>();
  float* pw = weight->data_ptr<float>();
  float* pout = output.data_ptr<float>();

  using Vec = at::vec::Vectorized<float>;
  constexpr int64_t vec_size = Vec::size();
  int64_t vec_end = (last_dim / vec_size) * vec_size;

  // Process each row (outer dimension)
  #pragma omp parallel for
  for (int64_t row = 0; row < outer_size; row++) {
    float* row_in = px + row * last_dim;
    float* row_out = pout + row * last_dim;

    // Compute sum of squares (variance without mean subtraction)
    float sum_sq = 0.0f;
    for (int64_t i = 0; i < vec_end; i += vec_size) {
      Vec vx = Vec::loadu(row_in + i);
      Vec vsq = vx * vx;
      // Horizontal sum within vector
      for (int64_t j = 0; j < vec_size; j++) {
        sum_sq += vsq[j];
      }
    }
    for (int64_t i = vec_end; i < last_dim; i++) {
      sum_sq += row_in[i] * row_in[i];
    }

    // Compute scaling factor: rsqrt(mean(x^2) + eps)
    float scale = 1.0f / std::sqrt(sum_sq / last_dim + eps);
    Vec vscale(scale);

    // Apply normalization and weight
    for (int64_t i = 0; i < vec_end; i += vec_size) {
      Vec vx = Vec::loadu(row_in + i);
      Vec vw = Vec::loadu(pw + i);
      Vec result = vx * vscale * vw;
      result.store(row_out + i);
    }
    for (int64_t i = vec_end; i < last_dim; i++) {
      row_out[i] = row_in[i] * scale * pw[i];
    }
  }

  return make_fused_tensor(std::move(output));
}

// ============================================================================
// Tensor shape/dtype fingerprinting (for cache keys)
// ============================================================================

// Takes a named list of SEXP tensors, returns a fingerprint string like:
//   "x=1500x384:f32,ln.weight=384:f32,..."
// One R->C++ call instead of 2N calls (N tensors * (shape + dtype)).
// [[Rcpp::export]]
SEXP cpp_tensor_shapes_key(SEXP tensor_list) {
  if (TYPEOF(tensor_list) != VECSXP) {
    Rf_error("Expected a list of tensors");
  }

  int n = Rf_length(tensor_list);
  SEXP names = Rf_getAttrib(tensor_list, R_NamesSymbol);

  std::string result;
  result.reserve(n * 32);  // Pre-allocate

  for (int i = 0; i < n; i++) {
    SEXP elem = VECTOR_ELT(tensor_list, i);
    at::Tensor* tptr = get_tensor_ptr(elem);
    if (!tptr) continue;

    // Append name
    if (names != R_NilValue && i < Rf_length(names)) {
      const char* nm = CHAR(STRING_ELT(names, i));
      result += nm;
    }
    result += '=';

    // Append shape: "1500x384"
    auto sizes = tptr->sizes();
    for (int64_t d = 0; d < tptr->dim(); d++) {
      if (d > 0) result += 'x';
      result += std::to_string(sizes[d]);
    }

    // Append dtype: ":f32"
    result += ':';
    auto dtype = tptr->scalar_type();
    switch (dtype) {
      case c10::ScalarType::Float:  result += "f32"; break;
      case c10::ScalarType::Double: result += "f64"; break;
      case c10::ScalarType::Half:   result += "f16"; break;
      case c10::ScalarType::BFloat16: result += "bf16"; break;
      case c10::ScalarType::Int:    result += "i32"; break;
      case c10::ScalarType::Long:   result += "i64"; break;
      case c10::ScalarType::Short:  result += "i16"; break;
      case c10::ScalarType::Byte:   result += "u8"; break;
      case c10::ScalarType::Bool:   result += "bool"; break;
      default: result += "unk"; break;
    }

    if (i < n - 1) result += ',';
  }

  return Rf_ScalarString(Rf_mkChar(result.c_str()));
}


