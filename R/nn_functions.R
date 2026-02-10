# ---- Namespace-level tensor functions ----

#' Compute sin of tensor
#' @param self A torch_tensor.
#' @export
torch_sin <- function(self) .Call(C_torch_sin, self)

#' Compute cos of tensor
#' @param self A torch_tensor.
#' @export
torch_cos <- function(self) .Call(C_torch_cos, self)

#' Compute tanh of tensor
#' @param self A torch_tensor.
#' @export
torch_tanh <- function(self) .Call(C_torch_tanh, self)

#' Compute reciprocal square root of tensor
#' @param self A torch_tensor.
#' @export
torch_rsqrt <- function(self) .Call(C_torch_rsqrt, self)

#' ReLU activation (namespace-level)
#' @param self A torch_tensor.
#' @export
torch_relu <- function(self) .Call(C_torch_relu, self)

# ---- NN functional API ----

#' ReLU activation (functional)
#' @param input A torch_tensor.
#' @export
nnf_relu <- function(input) .Call(C_torch_relu, input)

#' SiLU activation (swish)
#' @param self A torch_tensor.
#' @export
nnf_silu <- function(self) {
  .Call(C_nnf_silu, self)
}

#' GELU activation
#' @export
nnf_gelu <- function(self) {
  .Call(C_nnf_gelu, self)
}

#' Leaky ReLU activation
#' @export
nnf_leaky_relu <- function(self, negative_slope = 0.01) {
  .Call(C_nnf_leaky_relu, self, negative_slope)
}

#' ELU activation
#' @export
nnf_elu <- function(self, alpha = 1.0) {
  .Call(C_nnf_elu, self, alpha)
}

#' Softmax
#' @param self A torch_tensor.
#' @param dim Dimension to apply softmax over (1-based).
#' @export
nnf_softmax <- function(self, dim = -1L) {
  .Call(C_nnf_softmax, self, as.integer(dim))
}

#' Log-softmax
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
#' @export
torch_linear <- function(input, weight, bias = NULL) {
  .Call(C_torch_linear, input, weight, bias)
}

#' 1D convolution
#' @export
torch_conv1d <- function(input, weight, bias = NULL,
                         stride = 1L, padding = 0L,
                         dilation = 1L, groups = 1L) {
  .Call(C_torch_conv1d, input, weight, bias,
        as.integer(stride), as.integer(padding),
        as.integer(dilation), as.integer(groups))
}

#' Embedding lookup
#' @export
torch_embedding <- function(weight, indices) {
  .Call(C_torch_embedding, weight, indices)
}

# ---- Aliases for tracer (nnf_ -> torch_ expansion) ----

#' @rdname nnf_gelu
torch_gelu <- function(self) .Call(C_nnf_gelu, self)

#' @rdname nnf_silu
torch_silu <- function(self) .Call(C_nnf_silu, self)

#' @rdname nnf_softmax
torch_softmax <- function(self, dim = -1L, dtype = NULL) {
  .Call(C_nnf_softmax, self, as.integer(dim))
}

#' @rdname nnf_log_softmax
torch_log_softmax <- function(self, dim = -1L, dtype = NULL) {
  .Call(C_nnf_log_softmax, self, as.integer(dim))
}

#' @rdname nnf_layer_norm
torch_layer_norm <- function(input, normalized_shape, weight = NULL,
                             bias = NULL, eps = 1e-5) {
  .Call(C_nnf_layer_norm, input, as.integer(normalized_shape),
        weight, bias, eps)
}

# ---- Utilities ----

#' No-op gradient context manager
#'
#' Since Rtorch has no autograd, this simply evaluates the expression.
#' @param code Expression to evaluate.
#' @export
with_no_grad <- function(code) code
