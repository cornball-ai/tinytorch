# ---- Namespace-level tensor functions ----

#' Compute sin of tensor
#' @param self A torch_tensor.
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' torch_sin(torch_tensor(c(0, 3.14159 / 2)))
#' }
#' }
#' @return A `torch_tensor`.
#' @export
torch_sin <- function(self) C_torch_sin(self)

#' Compute cos of tensor
#' @param self A torch_tensor.
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' torch_cos(torch_tensor(c(0, 3.14159)))
#' }
#' }
#' @return A `torch_tensor`.
#' @export
torch_cos <- function(self) C_torch_cos(self)

#' Compute tanh of tensor
#' @param self A torch_tensor.
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' torch_tanh(torch_randn(c(2, 3)))
#' }
#' }
#' @return A `torch_tensor`.
#' @export
torch_tanh <- function(self) C_torch_tanh(self)

#' Compute reciprocal square root of tensor
#' @param self A torch_tensor.
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' torch_rsqrt(torch_tensor(c(1, 4, 9)))
#' }
#' }
#' @return A `torch_tensor`.
#' @export
torch_rsqrt <- function(self) C_torch_rsqrt(self)

#' ReLU activation (namespace-level)
#' @param self A torch_tensor.
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' torch_relu(torch_tensor(c(-1, 0, 1)))
#' }
#' }
#' @return A `torch_tensor`.
#' @export
torch_relu <- function(self) C_torch_relu(self)

# ---- NN functional API ----

#' ReLU activation (functional)
#' @param input A torch_tensor.
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' nnf_relu(torch_tensor(c(-1, 0, 1)))
#' }
#' }
#' @return A `torch_tensor`.
#' @export
nnf_relu <- function(input) C_torch_relu(input)

#' SiLU activation (swish)
#' @param self A torch_tensor.
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' nnf_silu(torch_randn(c(2, 3)))
#' }
#' }
#' @return A `torch_tensor`.
#' @export
nnf_silu <- function(self) {
  C_nnf_silu(self)
}

#' GELU activation
#' @param self A torch_tensor.
#' @param approximate Character, "none" or "tanh". Default "none".
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' nnf_gelu(torch_randn(c(2, 3)))
#' }
#' }
#' @return A `torch_tensor`.
#' @export
nnf_gelu <- function(self, approximate = "none") {
  C_nnf_gelu(self, approximate)
}

#' Leaky ReLU activation
#' @param self A torch_tensor.
#' @param negative_slope Negative slope coefficient. Default 0.01.
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' nnf_leaky_relu(torch_tensor(c(-1, 0, 1)))
#' }
#' }
#' @return A `torch_tensor`.
#' @export
nnf_leaky_relu <- function(self, negative_slope = 0.01) {
  C_nnf_leaky_relu(self, negative_slope)
}

#' ELU activation
#' @param self A torch_tensor.
#' @param alpha Scale for the negative factor. Default 1.0.
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' nnf_elu(torch_tensor(c(-1, 0, 1)))
#' }
#' }
#' @return A `torch_tensor`.
#' @export
nnf_elu <- function(self, alpha = 1.0) {
  C_nnf_elu(self, alpha)
}

#' Softmax
#' @param self A torch_tensor.
#' @param dim Dimension to apply softmax over (1-based).
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' nnf_softmax(torch_randn(c(2, 3)), dim = 2)
#' }
#' }
#' @return A `torch_tensor`.
#' @export
nnf_softmax <- function(self, dim = -1L) {
  C_nnf_softmax(self, as.integer(dim))
}

#' Log-softmax
#' @param self A torch_tensor.
#' @param dim Dimension to apply log-softmax over (1-based).
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' nnf_log_softmax(torch_randn(c(2, 3)), dim = 2)
#' }
#' }
#' @return A `torch_tensor`.
#' @export
nnf_log_softmax <- function(self, dim = -1L) {
  C_nnf_log_softmax(self, as.integer(dim))
}

#' Layer normalization
#' @param input A torch_tensor.
#' @param normalized_shape Integer vector of shape to normalize.
#' @param weight Optional weight tensor.
#' @param bias Optional bias tensor.
#' @param eps Small constant for numerical stability.
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' x <- torch_randn(c(2, 4))
#' nnf_layer_norm(x, 4L)
#' }
#' }
#' @return A `torch_tensor`.
#' @export
nnf_layer_norm <- function(input, normalized_shape, weight = NULL,
                           bias = NULL, eps = 1e-5) {
  C_nnf_layer_norm(input, as.integer(normalized_shape),
        weight, bias, eps)
}

# ---- Namespace-level torch functions ----

#' Linear transformation
#' @param input Input tensor.
#' @param weight Weight matrix.
#' @param bias Optional bias vector.
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' x <- torch_randn(c(2, 3))
#' w <- torch_randn(c(4, 3))
#' torch_linear(x, w)
#' }
#' }
#' @return A `torch_tensor`.
#' @export
torch_linear <- function(input, weight, bias = NULL) {
  C_torch_linear(input, weight, bias)
}

#' 1D convolution
#' @param input Input tensor of shape (N, C_in, L).
#' @param weight Filter tensor of shape (C_out, C_in/groups, kW).
#' @param bias Optional bias tensor of shape (C_out).
#' @param stride Stride of the convolution. Default 1.
#' @param padding Zero-padding added to both sides. Default 0.
#' @param dilation Spacing between kernel elements. Default 1.
#' @param groups Number of blocked connections. Default 1.
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' x <- torch_randn(c(1, 1, 10))
#' w <- torch_randn(c(1, 1, 3))
#' torch_conv1d(x, w)
#' }
#' }
#' @return A `torch_tensor`.
#' @export
torch_conv1d <- function(input, weight, bias = NULL,
                         stride = 1L, padding = 0L,
                         dilation = 1L, groups = 1L) {
  C_torch_conv1d(input, weight, bias,
        as.integer(stride), as.integer(padding),
        as.integer(dilation), as.integer(groups))
}

#' Embedding lookup
#' @param weight Embedding matrix tensor.
#' @param indices Integer tensor of indices.
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' emb <- torch_randn(c(10, 4))
#' idx <- torch_tensor(c(1L, 3L, 5L))
#' torch_embedding(emb, idx)
#' }
#' }
#' @return A `torch_tensor`.
#' @export
torch_embedding <- function(weight, indices) {
  C_torch_embedding(weight, indices)
}

# ---- Aliases for tracer (nnf_ -> torch_ expansion) ----

#' @rdname nnf_gelu
#' @param self A torch_tensor.
torch_gelu <- function(self) C_nnf_gelu(self, NULL)

#' @rdname nnf_silu
#' @param self A torch_tensor.
torch_silu <- function(self) C_nnf_silu(self)

#' @rdname nnf_softmax
#' @param self A torch_tensor.
#' @param dim Dimension to apply softmax over (1-based).
#' @param dtype Ignored; for API compatibility.
#' @return A `torch_tensor`.
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#'   # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
#' }
#' }
torch_softmax <- function(self, dim = -1L, dtype = NULL) {
  C_nnf_softmax(self, as.integer(dim))
}

#' @rdname nnf_log_softmax
#' @param self A torch_tensor.
#' @param dim Dimension to apply log-softmax over (1-based).
#' @param dtype Ignored; for API compatibility.
#' @return A `torch_tensor`.
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#'   # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
#' }
#' }
torch_log_softmax <- function(self, dim = -1L, dtype = NULL) {
  C_nnf_log_softmax(self, as.integer(dim))
}

#' @rdname nnf_layer_norm
#' @param input A torch_tensor.
#' @param normalized_shape Integer vector of shape to normalize.
#' @param weight Optional weight tensor.
#' @param bias Optional bias tensor.
#' @param eps Small constant for numerical stability.
#' @return A `torch_tensor`.
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#'   # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
#' }
#' }
torch_layer_norm <- function(input, normalized_shape, weight = NULL,
                             bias = NULL, eps = 1e-5) {
  C_nnf_layer_norm(input, as.integer(normalized_shape),
        weight, bias, eps)
}

# ---- Namespace exports for existing C++ unary ops ----

#' Absolute value
#' @param self A torch_tensor.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' torch_abs(torch_tensor(c(-1, -2, 3)))
#' }
#' }
torch_abs <- function(self) C_torch_abs(self)

#' Exponential
#' @param self A torch_tensor.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' torch_exp(torch_tensor(c(0, log(2))))
#' }
#' }
torch_exp <- function(self) C_torch_exp(self)

#' Natural logarithm
#' @param self A torch_tensor.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' a = torch_randn(c(5))
#' a
#' torch_log(a)
#' }
#' }
torch_log <- function(self) C_torch_log(self)

#' Square root
#' @param self A torch_tensor.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' a = torch_randn(c(4))
#' a
#' torch_sqrt(a)
#' }
#' }
torch_sqrt <- function(self) C_torch_sqrt(self)

#' Floor
#' @param self A torch_tensor.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' a = torch_randn(c(4))
#' a
#' torch_floor(a)
#' }
#' }
torch_floor <- function(self) C_torch_floor(self)

#' Power
#' @param self A torch_tensor or numeric scalar.
#' @param exponent A torch_tensor or numeric scalar.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' a = torch_randn(c(4))
#' a
#' torch_pow(a, 2)
#' exp <- torch_arange(1, 5)
#' a <- torch_arange(1, 5)
#' a
#' exp
#' torch_pow(a, exp)
#' 
#' 
#' exp <- torch_arange(1, 5)
#' base <- 2
#' torch_pow(base, exp)
#' }
#' }
torch_pow <- function(self, exponent) {
  if (!inherits(self, "torch_tensor") && inherits(exponent, "torch_tensor")) {
    # scalar ^ tensor
    C_torch_scalar_pow(self, exponent)
  } else if (is.numeric(exponent) && !inherits(exponent, "torch_tensor")) {
    # tensor ^ scalar
    C_torch_pow_scalar(self, exponent)
  } else {
    # tensor ^ tensor
    C_torch_pow(self, exponent)
  }
}

# ---- NN functional: pad ----

#' Pad a tensor
#'
#' Pads the input tensor using the specified padding mode.
#'
#' @param input A torch_tensor.
#' @param pad Integer vector of padding sizes.
#' @param mode Padding mode: "constant", "reflect", "replicate", or "circular".
#'   Default "constant".
#' @param value Fill value for constant padding. Default 0.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' x <- torch_randn(c(2, 3))
#' nnf_pad(x, c(1L, 1L))
#' }
#' }
nnf_pad <- function(input, pad, mode = "constant", value = 0) {
  C_nnf_pad(input, as.integer(pad), mode, as.double(value))
}

# ---- NN functional: interpolate ----

#' Interpolate a tensor
#'
#' Down/up samples the input to the given size or scale_factor.
#'
#' @param input A torch_tensor.
#' @param size Integer vector of output spatial sizes.
#' @param scale_factor Numeric multiplier for spatial size.
#' @param mode Interpolation mode: "nearest", "linear", "bilinear", "bicubic",
#'   "trilinear", or "area". Default "nearest".
#' @param align_corners Logical. If TRUE, aligns corner pixels. Default NULL.
#' @param recompute_scale_factor Logical. Ignored (for API compatibility).
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' x <- torch_randn(c(1, 1, 4, 4))
#' nnf_interpolate(x, size = c(8L, 8L), mode = "nearest")
#' }
#' }
nnf_interpolate <- function(input, size = NULL, scale_factor = NULL,
                            mode = "nearest", align_corners = NULL,
                            recompute_scale_factor = NULL) {
  size_int <- if (!is.null(size)) as.integer(size) else NULL
  scale_dbl <- if (!is.null(scale_factor)) as.double(scale_factor) else NULL
  ac <- if (!is.null(align_corners)) as.logical(align_corners) else NULL
  C_nnf_interpolate(input, size_int, scale_dbl, mode, ac)
}

# ---- NN functional: avg_pool1d ----

#' 1D average pooling
#'
#' Applies 1D average pooling over an input signal.
#'
#' @param input A torch_tensor of shape (N, C, L).
#' @param kernel_size Integer, size of the pooling window.
#' @param stride Integer, stride of the pooling window. Default kernel_size.
#' @param padding Integer, zero-padding on both sides. Default 0.
#' @param ceil_mode Logical. Use ceil instead of floor for output size. Default FALSE.
#' @param count_include_pad Logical. Include padding in average. Default TRUE.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' x <- torch_randn(c(1, 1, 8))
#' nnf_avg_pool1d(x, kernel_size = 2L)
#' }
#' }
nnf_avg_pool1d <- function(input, kernel_size, stride = kernel_size,
                           padding = 0L, ceil_mode = FALSE,
                           count_include_pad = TRUE) {
  C_nnf_avg_pool1d(input, as.integer(kernel_size),
        as.integer(stride), as.integer(padding),
        as.logical(ceil_mode), as.logical(count_include_pad))
}

# ---- NN functional: softplus ----

#' Softplus activation
#'
#' Applies the softplus function element-wise.
#'
#' @param input A torch_tensor.
#' @param beta Multiplier for the input. Default 1.0.
#' @param threshold Values above this revert to linear. Default 20.0.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' nnf_softplus(torch_randn(5))
#' }
#' }
nnf_softplus <- function(input, beta = 1.0, threshold = 20.0) {
  C_nnf_softplus(input, as.double(beta), as.double(threshold))
}

# ---- NN functional: normalize ----

#' Normalize a tensor along a dimension
#'
#' Performs Lp normalization along the given dimension.
#'
#' @param input A torch_tensor.
#' @param p Norm order. Default 2.
#' @param dim Dimension to normalize along. Default -1.
#' @param eps Small constant for numerical stability. Default 1e-12.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' nnf_normalize(torch_randn(c(2, 3)))
#' }
#' }
nnf_normalize <- function(input, p = 2, dim = -1L, eps = 1e-12) {
  C_nnf_normalize(input, as.double(p), as.integer(dim), as.double(eps))
}

# ---- NN functional: sigmoid ----

#' Sigmoid activation (functional)
#' @param input A torch_tensor.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' nnf_sigmoid(torch_randn(5))
#' }
#' }
nnf_sigmoid <- function(input) C_torch_sigmoid(input)

# ---- NN functional: tanh ----

#' Tanh activation (functional)
#' @param input A torch_tensor.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' nnf_tanh(torch_randn(5))
#' }
#' }
nnf_tanh <- function(input) C_torch_tanh(input)

# ---- Utilities ----

#' Disable gradient computation
#'
#' Evaluates the expression with gradient computation disabled.
#' @param code Expression to evaluate.
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' with_no_grad({
#'   x <- torch_randn(c(2, 3))
#'   torch_relu(x)
#' })
#' }
#' }
#' @return The result of evaluating `code`.
#' @export
with_no_grad <- function(code) {
  C_autograd_set_grad_mode(FALSE)
  on.exit(C_autograd_set_grad_mode(TRUE))
  force(code)
}

# ---- CUDA utilities ----

#' Set Number of CPU Threads
#'
#' Controls the number of threads used for intraop parallelism
#' (MKL, OpenBLAS, etc.) in libtorch operations.
#'
#' @param n Integer number of threads.
#' @return No return value, called for side effects.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' old <- torch_get_num_threads()
#' torch_set_num_threads(2L)
#' torch_set_num_threads(old)
#' }
#' }
torch_set_num_threads <- function(n) {
  invisible(C_torch_set_num_threads(as.integer(n)))
}

#' Get Number of CPU Threads
#'
#' @return Integer number of threads used for intraop parallelism.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' torch_get_num_threads()
#' }
#' }
torch_get_num_threads <- function() C_torch_get_num_threads()

#' Set Number of Interop Threads
#'
#' Controls the number of threads used for interop parallelism
#' (e.g., across independent operations) in libtorch.
#'
#' @param n Integer number of threads.
#' @return No return value, called for side effects.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' # May error if inter-op pool already initialized
#' try(torch_set_num_interop_threads(2L), silent = TRUE)
#' }
#' }
torch_set_num_interop_threads <- function(n) {
  invisible(C_torch_set_num_interop_threads(as.integer(n)))
}

#' Get Number of Interop Threads
#'
#' @return Integer number of interop threads.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' torch_get_num_interop_threads()
#' }
#' }
torch_get_num_interop_threads <- function() C_torch_get_num_interop_threads()

#' Check if CUDA is available
#' @return Logical scalar.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' cuda_is_available()
#' }
#' }
cuda_is_available <- function() C_cuda_is_available()

#' Get number of CUDA devices
#' @return Integer scalar.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' cuda_device_count()
#' }
#' }
cuda_device_count <- function() C_cuda_device_count()

#' Release cached CUDA memory
#'
#' Releases all unoccupied cached memory held by the CUDA allocator
#' so that it can be used by other GPU applications.
#' @return No return value, called for side effects.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' cuda_empty_cache()
#' }
#' }
cuda_empty_cache <- function() invisible(C_cuda_empty_cache())

#' CUDA memory info
#'
#' Returns free and total GPU memory in bytes for the current device,
#' as reported by the CUDA runtime (cudaMemGetInfo).
#' @return Named numeric vector with elements "free" and "total".
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' if (cuda_is_available()) cuda_mem_info()
#' }
#' }
cuda_mem_info <- function() {
    x <- C_cuda_mem_info()
    if (length(x) == 0) return(c(free = 0, total = 0))
    c(free = x[1], total = x[2])
}

#' CUDA memory statistics from libtorch caching allocator
#'
#' Returns allocated and reserved memory in bytes. "Allocated" is memory
#' actively used by tensors. "Reserved" is total memory held by the
#' caching allocator (allocated + cached free blocks).
#' @return Named numeric vector: allocated_current, allocated_peak,
#'   reserved_current, reserved_peak.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' if (cuda_is_available()) cuda_memory_stats()
#' }
#' }
cuda_memory_stats <- function() {
    x <- C_cuda_memory_stats()
    if (length(x) == 0) return(c(allocated_current = 0, allocated_peak = 0,
                                  reserved_current = 0, reserved_peak = 0))
    c(allocated_current = x[1], allocated_peak = x[2],
      reserved_current = x[3], reserved_peak = x[4])
}

#' No-op autocast context manager
#'
#' Since tinytorch is CPU-only, autocast is a no-op.
#' @param device_type Ignored.
#' @param dtype Ignored.
#' @param enabled Ignored.
#' @param code Expression to evaluate.
#' @return The result of evaluating `code`.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#'   # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
#' }
#' }
with_autocast <- function(device_type = "cpu", dtype = NULL,
                          enabled = TRUE, code) {
  code
}

#' Scaled Dot-Product Attention
#' @param query Query tensor (batch, heads, seq_q, head_dim).
#' @param key Key tensor (batch, heads, seq_k, head_dim).
#' @param value Value tensor (batch, heads, seq_k, head_dim).
#' @param attn_mask Optional attention mask tensor.
#' @param dropout_p Dropout probability (default 0).
#' @param is_causal Whether to apply causal masking (default FALSE).
#' @return Output tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' if (torch_is_installed()) {
#'   # Basic usage
#'   query <- torch_randn(2, 8, 10, 64)  # (batch, heads, seq_len, dim)
#'   key <- torch_randn(2, 8, 10, 64)
#'   value <- torch_randn(2, 8, 10, 64)
#' 
#'   output <- torch_scaled_dot_product_attention(query, key, value)
#' 
#'   # With causal masking (for autoregressive models)
#'   output <- torch_scaled_dot_product_attention(
#'     query, key, value,
#'     is_causal = TRUE
#'   )
#' 
#'   # With attention mask
#'   seq_len <- 10
#'   attn_mask <- torch_ones(seq_len, seq_len)
#'   attn_mask <- torch_tril(attn_mask)  # Lower triangular mask
#'   output <- torch_scaled_dot_product_attention(
#'     query, key, value,
#'     attn_mask = attn_mask
#'   )
#' }
#' 
#' }
#' }
torch_scaled_dot_product_attention <- function(query, key, value,
                                               attn_mask = NULL,
                                               dropout_p = 0.0,
                                               is_causal = FALSE) {
  # Handle list() sentinel (chatterbox passes list() for "no mask")
  if (is.list(attn_mask) && length(attn_mask) == 0) attn_mask <- NULL
  C_torch_sdpa(query, key, value, attn_mask,
        as.double(dropout_p), as.logical(is_causal))
}

#' Fused Transformer Decoder Layer Step
#'
#' Executes one pre-norm decoder layer entirely in C++ for incremental decoding
#' (seq_len=1 with KV cache). Replaces ~26 R-to-C++ crossings with a single call.
#'
#' @param x Input tensor (batch, 1, n_state).
#' @param weights List of 21 weight tensors in fixed order (see details).
#' @param self_cache_k Self-attention key cache (batch, n_head, seq_so_far, head_dim).
#' @param self_cache_v Self-attention value cache.
#' @param cross_cache_k Cross-attention key cache (batch, n_head, src_len, head_dim).
#' @param cross_cache_v Cross-attention value cache.
#' @param n_head Integer number of attention heads.
#' @return Named list: output (batch, 1, n_state), self_k (updated), self_v (updated).
#' @details
#' Weight list order (21 tensors):
#' \enumerate{
#'   \item attn_ln.weight, attn_ln.bias
#'   \item self_q.weight, self_q.bias
#'   \item self_k.weight (no bias)
#'   \item self_v.weight, self_v.bias
#'   \item self_out.weight, self_out.bias
#'   \item cross_attn_ln.weight, cross_attn_ln.bias
#'   \item cross_q.weight, cross_q.bias
#'   \item cross_out.weight, cross_out.bias
#'   \item mlp_ln.weight, mlp_ln.bias
#'   \item fc1.weight, fc1.bias
#'   \item fc2.weight, fc2.bias
#' }
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' # Low-level transformer op; see decoder_forward_step() for usage
#' }
#' }
transformer_decoder_layer_step <- function(x, weights, self_cache_k, self_cache_v,
                                           cross_cache_k, cross_cache_v, n_head) {
  stopifnot(is.list(weights), length(weights) == 21L)
  C_transformer_decoder_layer_step(x, weights,
        self_cache_k, self_cache_v, cross_cache_k, cross_cache_v,
        as.integer(n_head))
}

#' Fused Decoder Forward Step
#'
#' Executes the entire decoder forward pass in a single C++ call:
#' token embedding, positional embedding, all N decoder layers
#' (self-attn + cross-attn + MLP with KV cache), final layer norm,
#' logits projection, and argmax.
#'
#' @param token_ids Integer tensor (batch, seq_len) of 0-indexed token IDs.
#' @param global_weights List of 4 tensors: token_emb.weight, pos_emb.weight,
#'   final_ln.weight, final_ln.bias.
#' @param layer_weights List of N lists, each containing 21 weight tensors
#'   (same order as \code{\link{transformer_decoder_layer_step}}).
#' @param self_cache_k List of N tensors for self-attention key cache,
#'   or NULL for prefill.
#' @param self_cache_v List of N tensors for self-attention value cache,
#'   or NULL for prefill.
#' @param cross_cache_k List of N pre-computed cross-attention key tensors.
#' @param cross_cache_v List of N pre-computed cross-attention value tensors.
#' @param n_head Integer number of attention heads.
#' @param offset Integer position offset for positional embedding.
#' @return Named list: token_id (integer), self_cache_k, self_cache_v,
#'   cross_cache_k, cross_cache_v.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' # Specialized Whisper decoder step; see package vignettes
#' }
#' }
decoder_forward_step <- function(token_ids, global_weights, layer_weights,
                                 self_cache_k, self_cache_v,
                                 cross_cache_k, cross_cache_v,
                                 n_head, offset) {
  C_decoder_forward_step(token_ids, global_weights, layer_weights,
    self_cache_k, self_cache_v, cross_cache_k, cross_cache_v,
    as.integer(n_head), as.integer(offset))
}

#' Prepare Cross-Attention Caches
#'
#' Projects encoder output through each decoder layer's cross-attention
#' K/V weights in a single C++ call.
#'
#' @param encoder_output Encoder output tensor (batch, src_len, n_state).
#' @param cross_kv_weights List of N lists, each containing 3 tensors:
#'   cross_k.weight, cross_v.weight, cross_v.bias.
#' @param n_head Integer number of attention heads.
#' @return Named list with k and v, each a list of N tensors
#'   shaped (batch, n_head, src_len, head_dim).
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' # Specialized Whisper helper; see package vignettes
#' }
#' }
prepare_cross_caches <- function(encoder_output, cross_kv_weights, n_head) {
  C_prepare_cross_caches(encoder_output, cross_kv_weights, as.integer(n_head))
}

#' Fused Encoder Forward
#'
#' Executes the entire encoder forward pass in a single C++ call:
#' conv1+gelu, conv2+gelu, permute, positional embedding, all N
#' encoder layers (self-attention + MLP), and final layer norm.
#'
#' @param mel Input tensor (batch, n_mels, n_frames) mel spectrogram.
#' @param global_weights List of 7 tensors: conv1.weight, conv1.bias,
#'   conv2.weight, conv2.bias, positional_embedding, ln_post.weight, ln_post.bias.
#' @param layer_weights List of N lists, each containing 15 weight tensors
#'   (same order as \code{\link{transformer_encoder_layer}}).
#' @param n_head Integer number of attention heads.
#' @param n_ctx Integer maximum context length for sequence truncation.
#' @return Output tensor (batch, seq_len, n_state).
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' # Specialized Whisper encoder pipeline; see package vignettes
#' }
#' }
encoder_forward <- function(mel, global_weights, layer_weights, n_head, n_ctx) {
  C_encoder_forward(mel, global_weights, layer_weights,
    as.integer(n_head), as.integer(n_ctx))
}

#' Greedy Decode
#'
#' Runs the entire autoregressive decode loop in C++.
#' One boundary crossing for the full sequence -- no per-token R overhead.
#'
#' @param initial_tokens Integer vector of initial token IDs (0-indexed).
#' @param global_weights List of 4 tensors: token_emb.weight, pos_emb.weight,
#'   final_ln.weight, final_ln.bias.
#' @param layer_weights List of N lists, each containing 21 weight tensors.
#' @param cross_cache_k List of N pre-computed cross-attention key tensors.
#' @param cross_cache_v List of N pre-computed cross-attention value tensors.
#' @param n_head Integer number of attention heads.
#' @param max_length Maximum output sequence length.
#' @param eot_token End-of-text token ID (0-indexed).
#' @return Integer vector of generated token IDs (0-indexed).
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' # Specialized Whisper greedy decoder; see package vignettes
#' }
#' }
greedy_decode <- function(initial_tokens, global_weights, layer_weights,
                          cross_cache_k, cross_cache_v,
                          n_head, max_length, eot_token) {
  C_greedy_decode(as.integer(initial_tokens), global_weights, layer_weights,
    cross_cache_k, cross_cache_v,
    as.integer(n_head), as.integer(max_length), as.integer(eot_token))
}

#' Fused Transformer Encoder Layer
#'
#' Executes one pre-norm encoder layer entirely in C++.
#' Replaces ~20 R-to-C++ crossings with a single call.
#'
#' @param x Input tensor (batch, seq_len, n_state).
#' @param weights List of 15 weight tensors in fixed order (see details).
#' @param n_head Integer number of attention heads.
#' @return Output tensor (batch, seq_len, n_state).
#' @details
#' Weight list order (15 tensors):
#' \enumerate{
#'   \item attn_ln.weight, attn_ln.bias
#'   \item self_q.weight, self_q.bias
#'   \item self_k.weight (no bias)
#'   \item self_v.weight, self_v.bias
#'   \item self_out.weight, self_out.bias
#'   \item mlp_ln.weight, mlp_ln.bias
#'   \item fc1.weight, fc1.bias
#'   \item fc2.weight, fc2.bias
#' }
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' # Low-level transformer op; see encoder_forward() for full pipeline
#' }
#' }
transformer_encoder_layer <- function(x, weights, n_head) {
  stopifnot(is.list(weights), length(weights) == 15L)
  C_transformer_encoder_layer(x, weights, as.integer(n_head))
}
