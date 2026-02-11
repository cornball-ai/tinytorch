# ---- Namespace-level tensor functions ----

#' Compute sin of tensor
#' @param self A torch_tensor.
#' @examples
#' \donttest{
#' torch_sin(torch_tensor(c(0, 3.14159 / 2)))
#' }
#' @export
torch_sin <- function(self) .Call(C_torch_sin, self)

#' Compute cos of tensor
#' @param self A torch_tensor.
#' @examples
#' \donttest{
#' torch_cos(torch_tensor(c(0, 3.14159)))
#' }
#' @export
torch_cos <- function(self) .Call(C_torch_cos, self)

#' Compute tanh of tensor
#' @param self A torch_tensor.
#' @examples
#' \donttest{
#' torch_tanh(torch_randn(c(2, 3)))
#' }
#' @export
torch_tanh <- function(self) .Call(C_torch_tanh, self)

#' Compute reciprocal square root of tensor
#' @param self A torch_tensor.
#' @examples
#' \donttest{
#' torch_rsqrt(torch_tensor(c(1, 4, 9)))
#' }
#' @export
torch_rsqrt <- function(self) .Call(C_torch_rsqrt, self)

#' ReLU activation (namespace-level)
#' @param self A torch_tensor.
#' @examples
#' \donttest{
#' torch_relu(torch_tensor(c(-1, 0, 1)))
#' }
#' @export
torch_relu <- function(self) .Call(C_torch_relu, self)

# ---- NN functional API ----

#' ReLU activation (functional)
#' @param input A torch_tensor.
#' @examples
#' \donttest{
#' nnf_relu(torch_tensor(c(-1, 0, 1)))
#' }
#' @export
nnf_relu <- function(input) .Call(C_torch_relu, input)

#' SiLU activation (swish)
#' @param self A torch_tensor.
#' @examples
#' \donttest{
#' nnf_silu(torch_randn(c(2, 3)))
#' }
#' @export
nnf_silu <- function(self) {
  .Call(C_nnf_silu, self)
}

#' GELU activation
#' @param self A torch_tensor.
#' @examples
#' \donttest{
#' nnf_gelu(torch_randn(c(2, 3)))
#' }
#' @export
nnf_gelu <- function(self, approximate = "none") {
  .Call(C_nnf_gelu, self, approximate)
}

#' Leaky ReLU activation
#' @param self A torch_tensor.
#' @param negative_slope Negative slope coefficient. Default 0.01.
#' @examples
#' \donttest{
#' nnf_leaky_relu(torch_tensor(c(-1, 0, 1)))
#' }
#' @export
nnf_leaky_relu <- function(self, negative_slope = 0.01) {
  .Call(C_nnf_leaky_relu, self, negative_slope)
}

#' ELU activation
#' @param self A torch_tensor.
#' @param alpha Scale for the negative factor. Default 1.0.
#' @examples
#' \donttest{
#' nnf_elu(torch_tensor(c(-1, 0, 1)))
#' }
#' @export
nnf_elu <- function(self, alpha = 1.0) {
  .Call(C_nnf_elu, self, alpha)
}

#' Softmax
#' @param self A torch_tensor.
#' @param dim Dimension to apply softmax over (1-based).
#' @examples
#' \donttest{
#' nnf_softmax(torch_randn(c(2, 3)), dim = 2)
#' }
#' @export
nnf_softmax <- function(self, dim = -1L) {
  .Call(C_nnf_softmax, self, as.integer(dim))
}

#' Log-softmax
#' @param self A torch_tensor.
#' @param dim Dimension to apply log-softmax over (1-based).
#' @examples
#' \donttest{
#' nnf_log_softmax(torch_randn(c(2, 3)), dim = 2)
#' }
#' @export
nnf_log_softmax <- function(self, dim = -1L) {
  .Call(C_nnf_log_softmax, self, as.integer(dim))
}

#' Layer normalization
#' @param input A torch_tensor.
#' @param normalized_shape Integer vector of shape to normalize.
#' @param weight Optional weight tensor.
#' @param bias Optional bias tensor.
#' @param eps Small constant for numerical stability.
#' @examples
#' \donttest{
#' x <- torch_randn(c(2, 4))
#' nnf_layer_norm(x, 4L)
#' }
#' @export
nnf_layer_norm <- function(input, normalized_shape, weight = NULL,
                           bias = NULL, eps = 1e-5) {
  .Call(C_nnf_layer_norm, input, as.integer(normalized_shape),
        weight, bias, eps)
}

# ---- Namespace-level torch functions ----

#' Linear transformation
#' @param input Input tensor.
#' @param weight Weight matrix.
#' @param bias Optional bias vector.
#' @examples
#' \donttest{
#' x <- torch_randn(c(2, 3))
#' w <- torch_randn(c(4, 3))
#' torch_linear(x, w)
#' }
#' @export
torch_linear <- function(input, weight, bias = NULL) {
  .Call(C_torch_linear, input, weight, bias)
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
#' x <- torch_randn(c(1, 1, 10))
#' w <- torch_randn(c(1, 1, 3))
#' torch_conv1d(x, w)
#' }
#' @export
torch_conv1d <- function(input, weight, bias = NULL,
                         stride = 1L, padding = 0L,
                         dilation = 1L, groups = 1L) {
  .Call(C_torch_conv1d, input, weight, bias,
        as.integer(stride), as.integer(padding),
        as.integer(dilation), as.integer(groups))
}

#' Embedding lookup
#' @param weight Embedding matrix tensor.
#' @param indices Integer tensor of indices.
#' @examples
#' \donttest{
#' emb <- torch_randn(c(10, 4))
#' idx <- torch_tensor(c(1L, 3L, 5L))
#' torch_embedding(emb, idx)
#' }
#' @export
torch_embedding <- function(weight, indices) {
  .Call(C_torch_embedding, weight, indices)
}

# ---- Aliases for tracer (nnf_ -> torch_ expansion) ----

#' @rdname nnf_gelu
#' @param self A torch_tensor.
torch_gelu <- function(self) .Call(C_nnf_gelu, self)

#' @rdname nnf_silu
#' @param self A torch_tensor.
torch_silu <- function(self) .Call(C_nnf_silu, self)

#' @rdname nnf_softmax
#' @param self A torch_tensor.
#' @param dim Dimension to apply softmax over (1-based).
#' @param dtype Ignored; for API compatibility.
torch_softmax <- function(self, dim = -1L, dtype = NULL) {
  .Call(C_nnf_softmax, self, as.integer(dim))
}

#' @rdname nnf_log_softmax
#' @param self A torch_tensor.
#' @param dim Dimension to apply log-softmax over (1-based).
#' @param dtype Ignored; for API compatibility.
torch_log_softmax <- function(self, dim = -1L, dtype = NULL) {
  .Call(C_nnf_log_softmax, self, as.integer(dim))
}

#' @rdname nnf_layer_norm
#' @param input A torch_tensor.
#' @param normalized_shape Integer vector of shape to normalize.
#' @param weight Optional weight tensor.
#' @param bias Optional bias tensor.
#' @param eps Small constant for numerical stability.
torch_layer_norm <- function(input, normalized_shape, weight = NULL,
                             bias = NULL, eps = 1e-5) {
  .Call(C_nnf_layer_norm, input, as.integer(normalized_shape),
        weight, bias, eps)
}

# ---- Namespace exports for existing C++ unary ops ----

#' @export
torch_abs <- function(self) .Call(C_torch_abs, self)

#' @export
torch_exp <- function(self) .Call(C_torch_exp, self)

#' @export
torch_log <- function(self) .Call(C_torch_log, self)

#' @export
torch_sqrt <- function(self) .Call(C_torch_sqrt, self)

#' @export
torch_floor <- function(self) .Call(C_torch_floor, self)

#' @export
torch_pow <- function(self, exponent) {
  if (!inherits(self, "torch_tensor") && inherits(exponent, "torch_tensor")) {
    # scalar ^ tensor
    .Call(C_torch_scalar_pow, self, exponent)
  } else if (is.numeric(exponent) && !inherits(exponent, "torch_tensor")) {
    # tensor ^ scalar
    .Call(C_torch_pow_scalar, self, exponent)
  } else {
    # tensor ^ tensor
    .Call(C_torch_pow, self, exponent)
  }
}

# ---- NN functional: pad ----

#' @export
nnf_pad <- function(input, pad, mode = "constant", value = 0) {
  .Call(C_nnf_pad, input, as.integer(pad), mode, as.double(value))
}

# ---- NN functional: interpolate ----

#' @export
nnf_interpolate <- function(input, size = NULL, scale_factor = NULL,
                            mode = "nearest", align_corners = NULL,
                            recompute_scale_factor = NULL) {
  size_int <- if (!is.null(size)) as.integer(size) else NULL
  scale_dbl <- if (!is.null(scale_factor)) as.double(scale_factor) else NULL
  ac <- if (!is.null(align_corners)) as.logical(align_corners) else NULL
  .Call(C_nnf_interpolate, input, size_int, scale_dbl, mode, ac)
}

# ---- NN functional: avg_pool1d ----

#' @export
nnf_avg_pool1d <- function(input, kernel_size, stride = kernel_size,
                           padding = 0L, ceil_mode = FALSE,
                           count_include_pad = TRUE) {
  .Call(C_nnf_avg_pool1d, input, as.integer(kernel_size),
        as.integer(stride), as.integer(padding),
        as.logical(ceil_mode), as.logical(count_include_pad))
}

# ---- NN functional: softplus ----

#' @export
nnf_softplus <- function(input, beta = 1.0, threshold = 20.0) {
  .Call(C_nnf_softplus, input, as.double(beta), as.double(threshold))
}

# ---- NN functional: normalize ----

#' @export
nnf_normalize <- function(input, p = 2, dim = -1L, eps = 1e-12) {
  .Call(C_nnf_normalize, input, as.double(p), as.integer(dim), as.double(eps))
}

# ---- NN functional: sigmoid ----

#' @export
nnf_sigmoid <- function(input) .Call(C_torch_sigmoid, input)

# ---- NN functional: tanh ----

#' @export
nnf_tanh <- function(input) .Call(C_torch_tanh, input)

# ---- Utilities ----

#' No-op gradient context manager
#'
#' Since Rtorch has no autograd, this simply evaluates the expression.
#' @param code Expression to evaluate.
#' @examples
#' \donttest{
#' with_no_grad({
#'   x <- torch_randn(c(2, 3))
#'   torch_relu(x)
#' })
#' }
#' @export
with_no_grad <- function(code) code

#' No-op autocast context manager
#'
#' Since Rtorch is CPU-only, autocast is a no-op.
#' @param device_type Ignored.
#' @param dtype Ignored.
#' @param enabled Ignored.
#' @param code Expression to evaluate.
#' @export
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
torch_scaled_dot_product_attention <- function(query, key, value,
                                               attn_mask = NULL,
                                               dropout_p = 0.0,
                                               is_causal = FALSE) {
  # Handle list() sentinel (chatterbox passes list() for "no mask")
  if (is.list(attn_mask) && length(attn_mask) == 0) attn_mask <- NULL
  .Call(C_torch_sdpa, query, key, value, attn_mask,
        as.double(dropout_p), as.logical(is_causal))
}
