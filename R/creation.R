#' Create a tensor from R data
#'
#' @param data An R numeric, integer, or logical vector/matrix/array.
#' @param dtype Optional torch dtype (e.g. torch_float).
#' @param device Ignored for now (CPU only).
#' @param requires_grad Logical. Track gradients for this tensor. Default FALSE.
#' @return A torch_tensor object.
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' torch_tensor(c(1, 2, 3))
#' torch_tensor(matrix(1:6, 2, 3))
#' }
#' }
#' @export
torch_tensor <- function(data, dtype = NULL, device = NULL, requires_grad = FALSE) {
  if (inherits(data, "torch_tensor")) return(data)
  dtype_code <- .dtype_code(dtype)
  device_str <- if (!is.null(device)) as.character(device) else NULL
  t <- C_torch_tensor(data, dtype_code, device_str)
  if (requires_grad) C_tensor_requires_grad_(t, TRUE)
  t
}

#' Create a tensor of zeros
#' @param ... Dimensions (as separate integers or a single integer vector).
#' @param dtype Optional torch dtype.
#' @param device Ignored (CPU only).
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' torch_zeros(c(2, 3))
#' }
#' }
#' @return A `torch_tensor`.
#' @export
torch_zeros <- function(..., dtype = NULL, device = NULL) {
  size <- as.integer(c(...))
  dtype_code <- .dtype_code(dtype)
  device_str <- if (!is.null(device)) as.character(device) else NULL
  C_torch_zeros(size, dtype_code, device_str)
}

#' Create a tensor of ones
#' @param ... Dimensions (as separate integers or a single integer vector).
#' @param dtype Optional torch dtype.
#' @param device Ignored (CPU only).
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' torch_ones(c(2, 3))
#' }
#' }
#' @return A `torch_tensor`.
#' @export
torch_ones <- function(..., dtype = NULL, device = NULL) {
  size <- as.integer(c(...))
  dtype_code <- .dtype_code(dtype)
  device_str <- if (!is.null(device)) as.character(device) else NULL
  C_torch_ones(size, dtype_code, device_str)
}

#' Create a tensor with random normal values
#' @param ... Dimensions (as separate integers or a single integer vector).
#' @param dtype Optional torch dtype.
#' @param device Ignored (CPU only).
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' torch_randn(c(2, 3))
#' }
#' }
#' @return A `torch_tensor`.
#' @export
torch_randn <- function(..., dtype = NULL, device = NULL) {
  size <- as.integer(c(...))
  dtype_code <- .dtype_code(dtype)
  device_str <- if (!is.null(device)) as.character(device) else NULL
  C_torch_randn(size, dtype_code, device_str)
}

#' Create an uninitialized tensor with same shape/dtype as input
#' @param self A torch_tensor.
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' x <- torch_randn(c(2, 3))
#' torch_empty_like(x)
#' }
#' }
#' @return A `torch_tensor`.
#' @export
torch_empty_like <- function(self) {
  C_torch_empty_like(self)
}

#' Create an uninitialized tensor
#' @param ... Dimensions (as separate integers or a single integer vector).
#' @param dtype Optional torch dtype.
#' @param device Ignored (CPU only).
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' torch_empty(c(2, 3))
#' }
#' }
#' @return A `torch_tensor`.
#' @export
torch_empty <- function(..., dtype = NULL, device = NULL) {
  size <- as.integer(c(...))
  dtype_code <- .dtype_code(dtype)
  device_str <- if (!is.null(device)) as.character(device) else NULL
  C_torch_empty(size, dtype_code, device_str)
}

#' Create a tensor from raw bytes buffer
#' @param raw Raw vector of bytes.
#' @param shape Integer vector of dimensions.
#' @param dtype A torch_dtype.
#' @param device Ignored (CPU only).
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' # Construct a tensor from a raw buffer (advanced; see as_array() for the inverse)
#' }
#' }
torch_tensor_from_buffer <- function(raw, shape, dtype, device = NULL) {
  device_str <- if (!is.null(device)) as.character(device) else NULL
  C_torch_tensor_from_buffer(raw, as.integer(shape), .dtype_code(dtype), device_str)
}

#' Create a tensor with evenly spaced values
#' @param start Start value.
#' @param end End value (exclusive in PyTorch convention).
#' @param step Step between values. Default 1.
#' @param dtype Optional torch dtype.
#' @param device Ignored (CPU only).
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' torch_arange(start = 0, end = 5)
#' torch_arange(1, 4)
#' torch_arange(1, 2.5, 0.5)
#' }
#' }
torch_arange <- function(start, end, step = 1L, dtype = NULL, device = NULL) {
  dtype_code <- .dtype_code(dtype)
  device_str <- if (!is.null(device)) as.character(device) else NULL
  C_torch_arange(as.double(start), as.double(end),
        as.double(step), dtype_code, device_str)
}

#' Create a tensor filled with a scalar value
#' @param size Integer vector of dimensions.
#' @param fill_value Scalar fill value.
#' @param dtype Optional torch dtype.
#' @param device Ignored (CPU only).
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' torch_full(list(2, 3), 3.141592)
#' }
#' }
torch_full <- function(size, fill_value, dtype = NULL, device = NULL) {
  dtype_code <- .dtype_code(dtype)
  device_str <- if (!is.null(device)) as.character(device) else NULL
  C_torch_full(as.integer(size), as.double(fill_value), dtype_code, device_str)
}

#' Create a tensor with evenly spaced values between start and end
#' @param start Start value.
#' @param end End value.
#' @param steps Number of points. Default 100.
#' @param dtype Optional torch dtype.
#' @param device Ignored (CPU only).
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' torch_linspace(3, 10, steps=5)
#' torch_linspace(-10, 10, steps=5)
#' torch_linspace(start=-10, end=10, steps=5)
#' torch_linspace(start=-10, end=10, steps=1)
#' }
#' }
torch_linspace <- function(start, end, steps = 100L, dtype = NULL, device = NULL) {
  dtype_code <- .dtype_code(dtype)
  device_str <- if (!is.null(device)) as.character(device) else NULL
  C_torch_linspace(as.double(start), as.double(end),
        as.integer(steps), dtype_code, device_str)
}

#' Create a tensor of ones with same shape/dtype as input
#' @param self A torch_tensor.
#' @param dtype Optional dtype override.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' input = torch_empty(c(2, 3))
#' torch_ones_like(input)
#' }
#' }
torch_ones_like <- function(self, dtype = NULL) {
  C_torch_ones_like(self, .dtype_code(dtype))
}

#' Create a tensor of zeros with same shape/dtype as input
#' @param input A `torch_tensor`.
#' @param dtype Optional dtype override.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' input = torch_empty(c(2, 3))
#' torch_zeros_like(input)
#' }
#' }
torch_zeros_like <- function(input, dtype = NULL) {
  C_torch_zeros_like(input, .dtype_code(dtype))
}

#' Create a tensor of random normal values with same shape/dtype as input
#' @param input A `torch_tensor`.
#' @param dtype Optional dtype override.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' x <- torch_randn(c(2, 3))
#' torch_randn_like(x)
#' }
#' }
torch_randn_like <- function(input, dtype = NULL) {
  C_torch_randn_like(input, .dtype_code(dtype))
}

#' Concatenate tensors along a dimension
#' @param tensors A list of torch_tensors.
#' @param dim Dimension to concatenate along.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' x = torch_randn(c(2, 3))
#' x
#' torch_cat(list(x, x, x), 1)
#' torch_cat(list(x, x, x), 2)
#' }
#' }
torch_cat <- function(tensors, dim = 1L) {
  C_torch_cat(tensors, as.integer(dim))
}

#' Clamp tensor values to a range
#' @param self A torch_tensor.
#' @param min Minimum value (optional).
#' @param max Maximum value (optional).
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' a = torch_randn(c(4))
#' a
#' torch_clamp(a, min=-0.5, max=0.5)
#' 
#' 
#' a = torch_randn(c(4))
#' a
#' torch_clamp(a, min=0.5)
#' 
#' 
#' a = torch_randn(c(4))
#' a
#' torch_clamp(a, max=0.5)
#' }
#' }
torch_clamp <- function(self, min = NULL, max = NULL) {
  min_val <- if (!is.null(min)) as.double(min) else NULL
  max_val <- if (!is.null(max)) as.double(max) else NULL
  C_torch_clamp(self, min_val, max_val)
}

#' Element-wise selection based on condition
#' @param condition Boolean tensor.
#' @param self Tensor for TRUE elements.
#' @param other Tensor for FALSE elements.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' 
#' x = torch_randn(c(3, 2))
#' y = torch_ones(c(3, 2))
#' x
#' torch_where(x > 0, x, y)
#' 
#' 
#' 
#' 
#' }
#' }
torch_where <- function(condition, self, other) {
  C_torch_where(condition, self, other)
}

#' Sort tensor along a dimension
#' @param self A torch_tensor.
#' @param dim Dimension to sort along. Default -1.
#' @param descending Whether to sort in descending order. Default FALSE.
#' @return A list with values and indices tensors.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' x = torch_randn(c(3, 4))
#' out = torch_sort(x)
#' out
#' out = torch_sort(x, 1)
#' out
#' }
#' }
torch_sort <- function(self, dim = -1L, descending = FALSE) {
  C_torch_sort(self, as.integer(dim), as.logical(descending))
}

#' Flip tensor along given dimensions
#' @param self A torch_tensor.
#' @param dims Integer vector of dimensions to flip.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' x <- torch_arange(1, 8)$view(c(2, 2, 2))
#' x
#' torch_flip(x, c(1, 2))
#' }
#' }
torch_flip <- function(self, dims) {
  C_torch_flip(self, as.integer(dims))
}

#' Cumulative sum along a dimension
#' @param self A torch_tensor.
#' @param dim Dimension to compute cumsum along.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' a = torch_randn(c(10))
#' a
#' torch_cumsum(a, dim=1)
#' }
#' }
torch_cumsum <- function(self, dim) {
  C_torch_cumsum(self, as.integer(dim))
}

#' Element-wise maximum of two tensors
#' @param self A torch_tensor.
#' @param other A torch_tensor.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' a <- torch_tensor(c(1, 2, -1))
#' b <- torch_tensor(c(3, 0, 4))
#' torch_maximum(a, b)
#' }
#' }
torch_maximum <- function(self, other) {
  C_torch_maximum(self, other)
}

#' Draw samples from a multinomial distribution
#' @param self Probabilities tensor.
#' @param num_samples Number of samples per row.
#' @param replacement Whether to sample with replacement. Default FALSE.
#' @return A torch_tensor of indices.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' weights = torch_tensor(c(0, 10, 3, 0), dtype=torch_float()) # create a tensor of weights
#' torch_multinomial(weights, 2)
#' torch_multinomial(weights, 4, replacement=TRUE)
#' }
#' }
torch_multinomial <- function(self, num_samples, replacement = FALSE) {
  C_torch_multinomial(self, as.integer(num_samples), as.logical(replacement))
}

#' Outer product of two vectors
#' @param self First vector.
#' @param vec2 Second vector.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' v1 <- torch_arange(1., 5.)
#' v2 <- torch_arange(1., 4.)
#' torch_outer(v1, v2)
#' }
#' }
torch_outer <- function(self, vec2) {
  C_torch_outer(self, vec2)
}

#' Upper triangular part of a matrix
#' @param self A torch_tensor.
#' @param diagonal Offset from the main diagonal. Default 0.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' a = torch_randn(c(3, 3))
#' a
#' torch_triu(a)
#' torch_triu(a, diagonal=1)
#' torch_triu(a, diagonal=-1)
#' b = torch_randn(c(4, 6))
#' b
#' torch_triu(b, diagonal=1)
#' torch_triu(b, diagonal=-1)
#' }
#' }
torch_triu <- function(self, diagonal = 0L) {
  C_torch_triu(self, as.integer(diagonal))
}

#' Norm of a tensor
#' @param self A torch_tensor.
#' @param p Norm order. Default 2.
#' @param dim Optional dimension to reduce.
#' @param keepdim Whether to keep the reduced dimension.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' a <- torch_arange(1, 9, dtype = torch_float())
#' b <- a$reshape(list(3, 3))
#' torch_norm(a)
#' torch_norm(b)
#' torch_norm(a, Inf)
#' torch_norm(b, Inf)
#' 
#' }
#' }
torch_norm <- function(self, p = 2, dim = NULL, keepdim = FALSE) {
  C_torch_norm(self, as.double(p), dim, as.logical(keepdim))
}

#' Standard deviation of a tensor
#' @param self A torch_tensor.
#' @param dim Optional dimension to reduce.
#' @param keepdim Whether to keep the reduced dimension.
#' @param unbiased Logical. Use Bessel's correction. Default TRUE.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' a = torch_randn(c(1, 3))
#' a
#' torch_std(a)
#' 
#' 
#' a = torch_randn(c(4, 4))
#' a
#' torch_std(a, dim=1)
#' }
#' }
torch_std <- function(self, dim = NULL, keepdim = FALSE, unbiased = TRUE) {
  C_torch_std(self, dim, as.logical(keepdim), as.logical(unbiased))
}

# ---- Complex & signal processing ----

#' Create a complex tensor from real and imaginary parts
#' @param real Real part tensor.
#' @param imag Imaginary part tensor.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' real <- torch_tensor(c(1, 2), dtype=torch_float32())
#' imag <- torch_tensor(c(3, 4), dtype=torch_float32())
#' z <- torch_complex(real, imag)
#' z
#' z$dtype
#' }
#' }
torch_complex <- function(real, imag) {
  C_torch_complex(real, imag)
}

#' Extract real part of a complex tensor
#' @param self A torch_tensor.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#'   # Requires complex tensor support; see PyTorch docs
#' }
#' }
torch_real <- function(self) {
  C_torch_real(self)
}

#' Extract imaginary part of a complex tensor
#' @param self A torch_tensor.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#'   # Requires complex tensor support; see PyTorch docs
#' }
#' }
torch_imag <- function(self) {
  C_torch_imag(self)
}

#' Create a complex tensor from polar coordinates
#' @param abs Absolute value tensor.
#' @param angle Angle tensor.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' abs <- torch_tensor(c(1, 2), dtype=torch_float64())
#' angle <- torch_tensor(c(pi / 2, 5 * pi / 4), dtype=torch_float64())
#' z <- torch_polar(abs, angle)
#' z
#' }
#' }
torch_polar <- function(abs, angle) {
  C_torch_polar(abs, angle)
}

#' View a complex tensor as a real tensor with extra dimension
#' @param self A torch_tensor.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' 
#' if (FALSE) {
#' x <- torch_randn(4, dtype=torch_cfloat())
#' x
#' torch_view_as_real(x)
#' }
#' }
#' }
torch_view_as_real <- function(self) {
  C_torch_view_as_real(self)
}

#' Short-time Fourier Transform
#' @param input Input tensor.
#' @param n_fft FFT size.
#' @param hop_length Hop length. Default n_fft / 4.
#' @param win_length Window length. Default n_fft.
#' @param window Optional window tensor.
#' @param center Whether to pad input on both sides. Default TRUE.
#' @param normalized Whether to normalize. Default FALSE.
#' @param onesided Whether to return one-sided. Default TRUE.
#' @param return_complex Whether to return complex output. Default TRUE.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' x <- torch_randn(512)
#' torch_stft(x, n_fft = 64L, hop_length = 16L,
#'            window = torch_hann_window(64L), return_complex = TRUE)
#' }
#' }
torch_stft <- function(input, n_fft, hop_length = NULL, win_length = NULL,
                       window = NULL, center = TRUE, normalized = FALSE,
                       onesided = NULL, return_complex = TRUE) {
  hop <- if (!is.null(hop_length)) as.integer(hop_length) else as.integer(n_fft %/% 4L)
  winl <- if (!is.null(win_length)) as.integer(win_length) else as.integer(n_fft)
  os <- if (!is.null(onesided)) as.logical(onesided) else TRUE
  C_torch_stft(input, as.integer(n_fft), hop, winl,
        window, as.logical(center), as.logical(normalized),
        os, as.logical(return_complex))
}

#' Inverse Short-time Fourier Transform
#' @param input Input tensor.
#' @param n_fft FFT size.
#' @param hop_length Hop length. Default n_fft / 4.
#' @param win_length Window length. Default n_fft.
#' @param window Optional window tensor.
#' @param center Whether input was centered. Default TRUE.
#' @param normalized Whether input was normalized. Default FALSE.
#' @param onesided Whether input is one-sided. Default TRUE.
#' @param length Expected output length. Default NULL.
#' @param return_complex Whether to return complex output. Default FALSE.
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' x <- torch_randn(512)
#' win <- torch_hann_window(64L)
#' y <- torch_stft(x, n_fft = 64L, hop_length = 16L,
#'                 window = win, return_complex = TRUE)
#' torch_istft(y, n_fft = 64L, hop_length = 16L, window = win)
#' }
#' }
torch_istft <- function(input, n_fft, hop_length = NULL, win_length = NULL,
                        window = NULL, center = TRUE, normalized = FALSE,
                        onesided = TRUE, length = NULL,
                        return_complex = FALSE) {
  hop <- if (!is.null(hop_length)) as.integer(hop_length) else as.integer(n_fft %/% 4L)
  winl <- if (!is.null(win_length)) as.integer(win_length) else as.integer(n_fft)
  len <- if (!is.null(length)) as.integer(length) else NULL
  C_torch_istft(input, as.integer(n_fft), hop, winl,
        window, as.logical(center), as.logical(normalized),
        as.logical(onesided), len, as.logical(return_complex))
}

#' Create a Hann window tensor
#' @param window_length Length of the window.
#' @param periodic Whether the window is periodic. Default TRUE.
#' @param dtype Optional torch dtype.
#' @param device Ignored (CPU only).
#' @return A torch_tensor.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' torch_hann_window(16L)
#' }
#' }
torch_hann_window <- function(window_length, periodic = TRUE,
                              dtype = NULL, device = NULL) {
  dtype_code <- .dtype_code(dtype)
  device_str <- if (!is.null(device)) as.character(device) else NULL
  C_torch_hann_window(as.integer(window_length),
        as.logical(periodic), dtype_code, device_str)
}

#' Check if two tensors are element-wise close
#'
#' @param self A `torch_tensor`.
#' @param other A `torch_tensor`.
#' @param rtol Relative tolerance (default 1e-05).
#' @param atol Absolute tolerance (default 1e-08).
#' @return Logical scalar.
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' a <- torch_ones(c(2, 3))
#' b <- torch_ones(c(2, 3))
#' torch_allclose(a, b)
#' }
#' }
#' @export
torch_allclose <- function(self, other, rtol = 1e-05, atol = 1e-08) {
  diff <- (self - other)$abs()
  max_diff <- diff$max()$item()
  max_other <- other$abs()$max()$item()
  max_diff <= atol + rtol * max_other
}
