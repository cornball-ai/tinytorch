# Helper to wrap raw external pointer as torch_tensor
.wrap_tensor <- function(ptr) {
  class(ptr) <- "torch_tensor"
  ptr
}

# Ensure tensor is float32 for fused SIMD kernels
.ensure_float <- function(x) {
  if (unclass(x$dtype) != 6L) x$to(torch_float) else x
}

#' Fused relu
#' @param x A torch tensor.
#' @return A new torch tensor (float32).
#' @examples
#' \dontrun{
#' fused_relu(torch_randn(c(2, 3)))
#' }
#' @export
fused_relu <- function(x) {
  if (!inherits(x, "torch_tensor")) stop("Input must be a torch_tensor")
  .wrap_tensor(cpp_fused_relu(.ensure_float(x)))
}

#' Fused relu + sigmoid
#' @param x A torch tensor.
#' @return A new torch tensor (float32).
#' @examples
#' \dontrun{
#' fused_relu_sigmoid(torch_randn(c(2, 3)))
#' }
#' @export
fused_relu_sigmoid <- function(x) {
  if (!inherits(x, "torch_tensor")) stop("Input must be a torch_tensor")
  .wrap_tensor(cpp_fused_relu_sigmoid(.ensure_float(x)))
}

#' Fused relu + sigmoid + tanh
#' @param x A torch tensor.
#' @return A new torch tensor (float32).
#' @examples
#' \dontrun{
#' fused_relu_sigmoid_tanh(torch_randn(c(2, 3)))
#' }
#' @export
fused_relu_sigmoid_tanh <- function(x) {
  if (!inherits(x, "torch_tensor")) stop("Input must be a torch_tensor")
  .wrap_tensor(cpp_fused_relu_sigmoid_tanh(.ensure_float(x)))
}

#' Fused SiLU (Swish): x * sigmoid(x)
#' @param x A torch tensor.
#' @return A new torch tensor (float32).
#' @examples
#' \dontrun{
#' fused_silu(torch_randn(c(2, 3)))
#' }
#' @export
fused_silu <- function(x) {
  if (!inherits(x, "torch_tensor")) stop("Input must be a torch_tensor")
  .wrap_tensor(cpp_fused_silu(.ensure_float(x)))
}

#' Fused GELU activation
#' @param x A torch tensor.
#' @return A new torch tensor (float32).
#' @examples
#' \dontrun{
#' fused_gelu(torch_randn(c(2, 3)))
#' }
#' @export
fused_gelu <- function(x) {
  if (!inherits(x, "torch_tensor")) stop("Input must be a torch_tensor")
  .wrap_tensor(cpp_fused_gelu(.ensure_float(x)))
}

#' Fused sin and cos
#' @param x A torch tensor.
#' @return A list with sin and cos tensors (float32).
#' @examples
#' \dontrun{
#' fused_sincos(torch_randn(c(2, 3)))
#' }
#' @export
fused_sincos <- function(x) {
  if (!inherits(x, "torch_tensor")) stop("Input must be a torch_tensor")
  result <- cpp_fused_sincos(.ensure_float(x))
  list(
    sin = .wrap_tensor(result$sin),
    cos = .wrap_tensor(result$cos)
  )
}

#' Fused softcap: tanh(x/cap) * cap
#' @param x A torch tensor.
#' @param cap Scalar capping value.
#' @return A new torch tensor (float32).
#' @examples
#' \dontrun{
#' fused_softcap(torch_randn(c(2, 3)), 30.0)
#' }
#' @export
fused_softcap <- function(x, cap) {
  if (!inherits(x, "torch_tensor")) stop("Input must be a torch_tensor")
  .wrap_tensor(cpp_fused_softcap(.ensure_float(x), as.double(cap)))
}

#' Fused RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
#' @param x A torch tensor.
#' @param weight Weight tensor (same size as last dim of x).
#' @param eps Small constant for numerical stability.
#' @return A new torch tensor (float32).
#' @examples
#' \dontrun{
#' x <- torch_randn(c(2, 4))
#' w <- torch_ones(c(4))
#' fused_rmsnorm(x, w)
#' }
#' @export
fused_rmsnorm <- function(x, weight, eps = 1e-6) {
  if (!inherits(x, "torch_tensor") || !inherits(weight, "torch_tensor"))
    stop("Inputs must be torch_tensors")
  .wrap_tensor(cpp_fused_rmsnorm(.ensure_float(x), .ensure_float(weight),
                     as.double(eps)))
}
