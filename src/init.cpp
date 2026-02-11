#include "Rtorch.h"

// Forward declarations - tensor creation
extern "C" SEXP C_torch_tensor(SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_tensor_raw(SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_zeros(SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_ones(SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_randn(SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_empty_like(SEXP);
extern "C" SEXP C_torch_empty(SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_tensor_from_buffer(SEXP, SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_arange(SEXP, SEXP, SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_full(SEXP, SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_linspace(SEXP, SEXP, SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_ones_like(SEXP, SEXP);
extern "C" SEXP C_torch_zeros_like(SEXP, SEXP);
extern "C" SEXP C_torch_randn_like(SEXP, SEXP);

// Forward declarations - arithmetic ops
extern "C" SEXP C_torch_add(SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_sub(SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_mul(SEXP, SEXP);
extern "C" SEXP C_torch_div(SEXP, SEXP);
extern "C" SEXP C_torch_neg(SEXP);
extern "C" SEXP C_torch_logical_not(SEXP);

// Scalar arithmetic
extern "C" SEXP C_torch_add_scalar(SEXP, SEXP);
extern "C" SEXP C_torch_sub_scalar(SEXP, SEXP);
extern "C" SEXP C_torch_mul_scalar(SEXP, SEXP);
extern "C" SEXP C_torch_div_scalar(SEXP, SEXP);

// Forward declarations - linalg
extern "C" SEXP C_torch_matmul(SEXP, SEXP);
extern "C" SEXP C_torch_mm(SEXP, SEXP);
extern "C" SEXP C_torch_t(SEXP);

// Forward declarations - reduction
extern "C" SEXP C_torch_sum(SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_mean(SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_max(SEXP, SEXP);
extern "C" SEXP C_torch_min(SEXP, SEXP);

// Forward declarations - shape
extern "C" SEXP C_torch_reshape(SEXP, SEXP);
extern "C" SEXP C_torch_view(SEXP, SEXP);
extern "C" SEXP C_torch_squeeze(SEXP, SEXP);
extern "C" SEXP C_torch_unsqueeze(SEXP, SEXP);

// Forward declarations - unary ops
extern "C" SEXP C_torch_relu(SEXP);
extern "C" SEXP C_torch_sigmoid(SEXP);
extern "C" SEXP C_torch_tanh(SEXP);
extern "C" SEXP C_torch_exp(SEXP);
extern "C" SEXP C_torch_log(SEXP);
extern "C" SEXP C_torch_log2(SEXP);
extern "C" SEXP C_torch_log10(SEXP);
extern "C" SEXP C_torch_sqrt(SEXP);
extern "C" SEXP C_torch_abs(SEXP);
extern "C" SEXP C_torch_sign(SEXP);
extern "C" SEXP C_torch_floor(SEXP);
extern "C" SEXP C_torch_ceil(SEXP);
extern "C" SEXP C_torch_round(SEXP);
extern "C" SEXP C_torch_trunc(SEXP);
extern "C" SEXP C_torch_sin(SEXP);
extern "C" SEXP C_torch_cos(SEXP);
extern "C" SEXP C_torch_rsqrt(SEXP);
extern "C" SEXP C_torch_detach(SEXP);

// Forward declarations - binary ops
extern "C" SEXP C_torch_pow(SEXP, SEXP);
extern "C" SEXP C_torch_pow_scalar(SEXP, SEXP);
extern "C" SEXP C_torch_scalar_pow(SEXP, SEXP);
extern "C" SEXP C_torch_remainder(SEXP, SEXP);
extern "C" SEXP C_torch_remainder_scalar(SEXP, SEXP);
extern "C" SEXP C_torch_floor_divide(SEXP, SEXP);
extern "C" SEXP C_torch_floor_divide_scalar(SEXP, SEXP);

// Forward declarations - comparison ops
extern "C" SEXP C_torch_eq(SEXP, SEXP);
extern "C" SEXP C_torch_eq_scalar(SEXP, SEXP);
extern "C" SEXP C_torch_ne(SEXP, SEXP);
extern "C" SEXP C_torch_ne_scalar(SEXP, SEXP);
extern "C" SEXP C_torch_lt(SEXP, SEXP);
extern "C" SEXP C_torch_lt_scalar(SEXP, SEXP);
extern "C" SEXP C_torch_le(SEXP, SEXP);
extern "C" SEXP C_torch_le_scalar(SEXP, SEXP);
extern "C" SEXP C_torch_gt(SEXP, SEXP);
extern "C" SEXP C_torch_gt_scalar(SEXP, SEXP);
extern "C" SEXP C_torch_ge(SEXP, SEXP);
extern "C" SEXP C_torch_ge_scalar(SEXP, SEXP);

// Forward declarations - additional linalg/shape
extern "C" SEXP C_torch_bmm(SEXP, SEXP);
extern "C" SEXP C_torch_transpose(SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_flatten(SEXP, SEXP, SEXP);

// Forward declarations - utility
extern "C" SEXP C_torch_clone(SEXP);
extern "C" SEXP C_torch_contiguous(SEXP);
extern "C" SEXP C_torch_to_dtype(SEXP, SEXP);
extern "C" SEXP C_torch_item(SEXP);

// Forward declarations - new tensor ops
extern "C" SEXP C_torch_cat(SEXP, SEXP);
extern "C" SEXP C_torch_clamp(SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_where(SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_sort(SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_flip(SEXP, SEXP);
extern "C" SEXP C_torch_cumsum(SEXP, SEXP);
extern "C" SEXP C_torch_maximum(SEXP, SEXP);
extern "C" SEXP C_torch_multinomial(SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_outer(SEXP, SEXP);
extern "C" SEXP C_torch_triu(SEXP, SEXP);
extern "C" SEXP C_torch_norm(SEXP, SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_std(SEXP, SEXP, SEXP, SEXP);

// Forward declarations - complex & signal processing
extern "C" SEXP C_torch_complex(SEXP, SEXP);
extern "C" SEXP C_torch_real(SEXP);
extern "C" SEXP C_torch_imag(SEXP);
extern "C" SEXP C_torch_polar(SEXP, SEXP);
extern "C" SEXP C_torch_view_as_real(SEXP);
extern "C" SEXP C_torch_stft(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_istft(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_hann_window(SEXP, SEXP, SEXP, SEXP);

// Forward declarations - tensor methods
extern "C" SEXP C_torch_permute(SEXP, SEXP);
extern "C" SEXP C_torch_expand(SEXP, SEXP);
extern "C" SEXP C_torch_gather(SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_masked_fill(SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_masked_fill_(SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_copy_(SEXP, SEXP);
extern "C" SEXP C_torch_normal_(SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_uniform_(SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_zero_(SEXP);
extern "C" SEXP C_torch_fill_(SEXP, SEXP);
extern "C" SEXP C_torch_repeat(SEXP, SEXP);
extern "C" SEXP C_torch_repeat_interleave(SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_index_select(SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_narrow(SEXP, SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_scatter_(SEXP, SEXP, SEXP, SEXP);

// Forward declarations - NN functions
extern "C" SEXP C_nnf_silu(SEXP);
extern "C" SEXP C_nnf_gelu(SEXP, SEXP);
extern "C" SEXP C_nnf_leaky_relu(SEXP, SEXP);
extern "C" SEXP C_nnf_elu(SEXP, SEXP);
extern "C" SEXP C_nnf_softmax(SEXP, SEXP);
extern "C" SEXP C_nnf_log_softmax(SEXP, SEXP);
extern "C" SEXP C_nnf_layer_norm(SEXP, SEXP, SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_linear(SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_conv1d(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_embedding(SEXP, SEXP);
extern "C" SEXP C_torch_conv_transpose1d(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_conv2d(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_batch_norm(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_lstm(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern "C" SEXP C_nnf_pad(SEXP, SEXP, SEXP, SEXP);
extern "C" SEXP C_nnf_interpolate(SEXP, SEXP, SEXP, SEXP, SEXP);
extern "C" SEXP C_nnf_avg_pool1d(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern "C" SEXP C_nnf_softplus(SEXP, SEXP, SEXP);
extern "C" SEXP C_nnf_normalize(SEXP, SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_sdpa(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

// Forward declarations - fused kernels
extern "C" SEXP cpp_fused_relu(SEXP);
extern "C" SEXP cpp_fused_relu_sigmoid(SEXP);
extern "C" SEXP cpp_fused_relu_sigmoid_tanh(SEXP);
extern "C" SEXP cpp_fused_silu(SEXP);
extern "C" SEXP cpp_fused_gelu(SEXP);
extern "C" SEXP cpp_fused_sincos(SEXP);
extern "C" SEXP cpp_fused_softcap(SEXP, SEXP);
extern "C" SEXP cpp_fused_rmsnorm(SEXP, SEXP, SEXP);
extern "C" SEXP cpp_tensor_shapes_key(SEXP);

// Forward declarations - indexing
extern "C" SEXP C_torch_index(SEXP, SEXP, SEXP);
extern "C" SEXP C_torch_index_put(SEXP, SEXP, SEXP);

// Forward declarations - conversion & properties
extern "C" SEXP C_as_array(SEXP);
extern "C" SEXP C_tensor_shape(SEXP);
extern "C" SEXP C_tensor_dtype(SEXP);
extern "C" SEXP C_tensor_device(SEXP);
extern "C" SEXP C_tensor_ndim(SEXP);
extern "C" SEXP C_tensor_numel(SEXP);
extern "C" SEXP C_tensor_requires_grad(SEXP);
extern "C" SEXP C_tensor_print(SEXP);

// Forward declarations - device transfer & CUDA
extern "C" SEXP C_tensor_to_device(SEXP, SEXP);
extern "C" SEXP C_tensor_to_dtype_device(SEXP, SEXP, SEXP);
extern "C" SEXP C_cuda_is_available();
extern "C" SEXP C_cuda_device_count();
extern "C" SEXP C_cuda_empty_cache();
extern "C" SEXP C_cuda_mem_info();
extern "C" SEXP C_cuda_memory_stats();

static const R_CallMethodDef CallEntries[] = {
    // Creation
    {"C_torch_tensor",      (DL_FUNC) &C_torch_tensor,      3},
    {"C_torch_tensor_raw",  (DL_FUNC) &C_torch_tensor_raw,  3},
    {"C_torch_zeros",       (DL_FUNC) &C_torch_zeros,       3},
    {"C_torch_ones",        (DL_FUNC) &C_torch_ones,        3},
    {"C_torch_randn",       (DL_FUNC) &C_torch_randn,       3},
    {"C_torch_empty_like",  (DL_FUNC) &C_torch_empty_like,  1},
    {"C_torch_empty",       (DL_FUNC) &C_torch_empty,       3},
    {"C_torch_tensor_from_buffer", (DL_FUNC) &C_torch_tensor_from_buffer, 4},
    {"C_torch_arange",      (DL_FUNC) &C_torch_arange,      5},
    {"C_torch_full",        (DL_FUNC) &C_torch_full,        4},
    {"C_torch_linspace",    (DL_FUNC) &C_torch_linspace,    5},
    {"C_torch_ones_like",   (DL_FUNC) &C_torch_ones_like,   2},
    {"C_torch_zeros_like",  (DL_FUNC) &C_torch_zeros_like,  2},
    {"C_torch_randn_like",  (DL_FUNC) &C_torch_randn_like,  2},

    // Arithmetic
    {"C_torch_add",         (DL_FUNC) &C_torch_add,         3},
    {"C_torch_sub",         (DL_FUNC) &C_torch_sub,         3},
    {"C_torch_mul",         (DL_FUNC) &C_torch_mul,         2},
    {"C_torch_div",         (DL_FUNC) &C_torch_div,         2},
    {"C_torch_neg",         (DL_FUNC) &C_torch_neg,         1},
    {"C_torch_logical_not", (DL_FUNC) &C_torch_logical_not, 1},

    // Scalar arithmetic
    {"C_torch_add_scalar",  (DL_FUNC) &C_torch_add_scalar,  2},
    {"C_torch_sub_scalar",  (DL_FUNC) &C_torch_sub_scalar,  2},
    {"C_torch_mul_scalar",  (DL_FUNC) &C_torch_mul_scalar,  2},
    {"C_torch_div_scalar",  (DL_FUNC) &C_torch_div_scalar,  2},

    // Linear algebra
    {"C_torch_matmul",      (DL_FUNC) &C_torch_matmul,      2},
    {"C_torch_mm",          (DL_FUNC) &C_torch_mm,          2},
    {"C_torch_t",           (DL_FUNC) &C_torch_t,           1},

    // Reduction
    {"C_torch_sum",         (DL_FUNC) &C_torch_sum,         3},
    {"C_torch_mean",        (DL_FUNC) &C_torch_mean,        3},
    {"C_torch_max",         (DL_FUNC) &C_torch_max,         2},
    {"C_torch_min",         (DL_FUNC) &C_torch_min,         2},

    // Shape
    {"C_torch_reshape",     (DL_FUNC) &C_torch_reshape,     2},
    {"C_torch_view",        (DL_FUNC) &C_torch_view,        2},
    {"C_torch_squeeze",     (DL_FUNC) &C_torch_squeeze,     2},
    {"C_torch_unsqueeze",   (DL_FUNC) &C_torch_unsqueeze,   2},

    // Unary ops
    {"C_torch_relu",        (DL_FUNC) &C_torch_relu,        1},
    {"C_torch_sigmoid",     (DL_FUNC) &C_torch_sigmoid,     1},
    {"C_torch_tanh",        (DL_FUNC) &C_torch_tanh,        1},
    {"C_torch_exp",         (DL_FUNC) &C_torch_exp,         1},
    {"C_torch_log",         (DL_FUNC) &C_torch_log,         1},
    {"C_torch_log2",        (DL_FUNC) &C_torch_log2,        1},
    {"C_torch_log10",       (DL_FUNC) &C_torch_log10,       1},
    {"C_torch_sqrt",        (DL_FUNC) &C_torch_sqrt,        1},
    {"C_torch_abs",         (DL_FUNC) &C_torch_abs,         1},
    {"C_torch_sign",        (DL_FUNC) &C_torch_sign,        1},
    {"C_torch_floor",       (DL_FUNC) &C_torch_floor,       1},
    {"C_torch_ceil",        (DL_FUNC) &C_torch_ceil,        1},
    {"C_torch_round",       (DL_FUNC) &C_torch_round,       1},
    {"C_torch_trunc",       (DL_FUNC) &C_torch_trunc,       1},
    {"C_torch_sin",         (DL_FUNC) &C_torch_sin,         1},
    {"C_torch_cos",         (DL_FUNC) &C_torch_cos,         1},
    {"C_torch_rsqrt",       (DL_FUNC) &C_torch_rsqrt,       1},
    {"C_torch_detach",      (DL_FUNC) &C_torch_detach,      1},

    // Binary ops
    {"C_torch_pow",                 (DL_FUNC) &C_torch_pow,                 2},
    {"C_torch_pow_scalar",          (DL_FUNC) &C_torch_pow_scalar,          2},
    {"C_torch_scalar_pow",          (DL_FUNC) &C_torch_scalar_pow,          2},
    {"C_torch_remainder",           (DL_FUNC) &C_torch_remainder,           2},
    {"C_torch_remainder_scalar",    (DL_FUNC) &C_torch_remainder_scalar,    2},
    {"C_torch_floor_divide",        (DL_FUNC) &C_torch_floor_divide,        2},
    {"C_torch_floor_divide_scalar", (DL_FUNC) &C_torch_floor_divide_scalar, 2},

    // Comparison ops
    {"C_torch_eq",          (DL_FUNC) &C_torch_eq,          2},
    {"C_torch_eq_scalar",   (DL_FUNC) &C_torch_eq_scalar,   2},
    {"C_torch_ne",          (DL_FUNC) &C_torch_ne,          2},
    {"C_torch_ne_scalar",   (DL_FUNC) &C_torch_ne_scalar,   2},
    {"C_torch_lt",          (DL_FUNC) &C_torch_lt,          2},
    {"C_torch_lt_scalar",   (DL_FUNC) &C_torch_lt_scalar,   2},
    {"C_torch_le",          (DL_FUNC) &C_torch_le,          2},
    {"C_torch_le_scalar",   (DL_FUNC) &C_torch_le_scalar,   2},
    {"C_torch_gt",          (DL_FUNC) &C_torch_gt,          2},
    {"C_torch_gt_scalar",   (DL_FUNC) &C_torch_gt_scalar,   2},
    {"C_torch_ge",          (DL_FUNC) &C_torch_ge,          2},
    {"C_torch_ge_scalar",   (DL_FUNC) &C_torch_ge_scalar,   2},

    // Additional linalg/shape
    {"C_torch_bmm",         (DL_FUNC) &C_torch_bmm,         2},
    {"C_torch_transpose",   (DL_FUNC) &C_torch_transpose,   3},
    {"C_torch_flatten",     (DL_FUNC) &C_torch_flatten,     3},

    // Utility
    {"C_torch_clone",       (DL_FUNC) &C_torch_clone,       1},
    {"C_torch_contiguous",  (DL_FUNC) &C_torch_contiguous,  1},
    {"C_torch_to_dtype",    (DL_FUNC) &C_torch_to_dtype,    2},
    {"C_torch_item",        (DL_FUNC) &C_torch_item,        1},

    // New tensor ops
    {"C_torch_cat",             (DL_FUNC) &C_torch_cat,             2},
    {"C_torch_clamp",           (DL_FUNC) &C_torch_clamp,           3},
    {"C_torch_where",           (DL_FUNC) &C_torch_where,           3},
    {"C_torch_sort",            (DL_FUNC) &C_torch_sort,            3},
    {"C_torch_flip",            (DL_FUNC) &C_torch_flip,            2},
    {"C_torch_cumsum",          (DL_FUNC) &C_torch_cumsum,          2},
    {"C_torch_maximum",         (DL_FUNC) &C_torch_maximum,         2},
    {"C_torch_multinomial",     (DL_FUNC) &C_torch_multinomial,     3},
    {"C_torch_outer",           (DL_FUNC) &C_torch_outer,           2},
    {"C_torch_triu",            (DL_FUNC) &C_torch_triu,            2},
    {"C_torch_norm",            (DL_FUNC) &C_torch_norm,            4},
    {"C_torch_std",             (DL_FUNC) &C_torch_std,             4},

    // Complex & signal processing
    {"C_torch_complex",         (DL_FUNC) &C_torch_complex,         2},
    {"C_torch_real",            (DL_FUNC) &C_torch_real,            1},
    {"C_torch_imag",            (DL_FUNC) &C_torch_imag,            1},
    {"C_torch_polar",           (DL_FUNC) &C_torch_polar,           2},
    {"C_torch_view_as_real",    (DL_FUNC) &C_torch_view_as_real,    1},
    {"C_torch_stft",            (DL_FUNC) &C_torch_stft,            9},
    {"C_torch_istft",           (DL_FUNC) &C_torch_istft,           10},
    {"C_torch_hann_window",     (DL_FUNC) &C_torch_hann_window,     4},

    // Tensor methods
    {"C_torch_permute",             (DL_FUNC) &C_torch_permute,             2},
    {"C_torch_expand",              (DL_FUNC) &C_torch_expand,              2},
    {"C_torch_gather",              (DL_FUNC) &C_torch_gather,              3},
    {"C_torch_masked_fill",         (DL_FUNC) &C_torch_masked_fill,         3},
    {"C_torch_masked_fill_",        (DL_FUNC) &C_torch_masked_fill_,        3},
    {"C_torch_copy_",               (DL_FUNC) &C_torch_copy_,               2},
    {"C_torch_normal_",             (DL_FUNC) &C_torch_normal_,             3},
    {"C_torch_uniform_",            (DL_FUNC) &C_torch_uniform_,            3},
    {"C_torch_zero_",               (DL_FUNC) &C_torch_zero_,               1},
    {"C_torch_fill_",               (DL_FUNC) &C_torch_fill_,               2},
    {"C_torch_repeat",              (DL_FUNC) &C_torch_repeat,              2},
    {"C_torch_repeat_interleave",   (DL_FUNC) &C_torch_repeat_interleave,   3},
    {"C_torch_index_select",        (DL_FUNC) &C_torch_index_select,        3},
    {"C_torch_narrow",              (DL_FUNC) &C_torch_narrow,              4},
    {"C_torch_scatter_",            (DL_FUNC) &C_torch_scatter_,            4},

    // NN functions
    {"C_nnf_silu",              (DL_FUNC) &C_nnf_silu,              1},
    {"C_nnf_gelu",              (DL_FUNC) &C_nnf_gelu,              2},
    {"C_nnf_leaky_relu",        (DL_FUNC) &C_nnf_leaky_relu,        2},
    {"C_nnf_elu",               (DL_FUNC) &C_nnf_elu,               2},
    {"C_nnf_softmax",           (DL_FUNC) &C_nnf_softmax,           2},
    {"C_nnf_log_softmax",       (DL_FUNC) &C_nnf_log_softmax,       2},
    {"C_nnf_layer_norm",        (DL_FUNC) &C_nnf_layer_norm,        5},
    {"C_torch_linear",          (DL_FUNC) &C_torch_linear,          3},
    {"C_torch_conv1d",          (DL_FUNC) &C_torch_conv1d,          7},
    {"C_torch_embedding",       (DL_FUNC) &C_torch_embedding,       2},
    {"C_torch_conv_transpose1d", (DL_FUNC) &C_torch_conv_transpose1d, 8},
    {"C_torch_conv2d",          (DL_FUNC) &C_torch_conv2d,          7},
    {"C_torch_batch_norm",      (DL_FUNC) &C_torch_batch_norm,      9},
    {"C_torch_lstm",            (DL_FUNC) &C_torch_lstm,            8},
    {"C_nnf_pad",               (DL_FUNC) &C_nnf_pad,               4},
    {"C_nnf_interpolate",       (DL_FUNC) &C_nnf_interpolate,       5},
    {"C_nnf_avg_pool1d",        (DL_FUNC) &C_nnf_avg_pool1d,        6},
    {"C_nnf_softplus",          (DL_FUNC) &C_nnf_softplus,          3},
    {"C_nnf_normalize",         (DL_FUNC) &C_nnf_normalize,         4},
    {"C_torch_sdpa",            (DL_FUNC) &C_torch_sdpa,            6},

    // Indexing
    {"C_torch_index",           (DL_FUNC) &C_torch_index,           3},
    {"C_torch_index_put",       (DL_FUNC) &C_torch_index_put,       3},

    // Fused kernels
    {"cpp_fused_relu",              (DL_FUNC) &cpp_fused_relu,              1},
    {"cpp_fused_relu_sigmoid",      (DL_FUNC) &cpp_fused_relu_sigmoid,      1},
    {"cpp_fused_relu_sigmoid_tanh", (DL_FUNC) &cpp_fused_relu_sigmoid_tanh, 1},
    {"cpp_fused_silu",              (DL_FUNC) &cpp_fused_silu,              1},
    {"cpp_fused_gelu",              (DL_FUNC) &cpp_fused_gelu,              1},
    {"cpp_fused_sincos",            (DL_FUNC) &cpp_fused_sincos,            1},
    {"cpp_fused_softcap",           (DL_FUNC) &cpp_fused_softcap,           2},
    {"cpp_fused_rmsnorm",           (DL_FUNC) &cpp_fused_rmsnorm,           3},
    {"cpp_tensor_shapes_key",       (DL_FUNC) &cpp_tensor_shapes_key,       1},

    // Conversion & properties
    {"C_as_array",              (DL_FUNC) &C_as_array,              1},
    {"C_tensor_shape",          (DL_FUNC) &C_tensor_shape,          1},
    {"C_tensor_dtype",          (DL_FUNC) &C_tensor_dtype,          1},
    {"C_tensor_device",         (DL_FUNC) &C_tensor_device,         1},
    {"C_tensor_ndim",           (DL_FUNC) &C_tensor_ndim,           1},
    {"C_tensor_numel",          (DL_FUNC) &C_tensor_numel,          1},
    {"C_tensor_requires_grad",  (DL_FUNC) &C_tensor_requires_grad,  1},
    {"C_tensor_print",          (DL_FUNC) &C_tensor_print,          1},

    // Device transfer & CUDA
    {"C_tensor_to_device",      (DL_FUNC) &C_tensor_to_device,      2},
    {"C_tensor_to_dtype_device", (DL_FUNC) &C_tensor_to_dtype_device, 3},
    {"C_cuda_is_available",     (DL_FUNC) &C_cuda_is_available,     0},
    {"C_cuda_device_count",     (DL_FUNC) &C_cuda_device_count,     0},
    {"C_cuda_empty_cache",      (DL_FUNC) &C_cuda_empty_cache,      0},
    {"C_cuda_mem_info",         (DL_FUNC) &C_cuda_mem_info,         0},
    {"C_cuda_memory_stats",     (DL_FUNC) &C_cuda_memory_stats,     0},

    {NULL, NULL, 0}
};

extern "C" void R_init_Rtorch(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
