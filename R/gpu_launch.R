# GPU kernel launch via CUDA driver API
#
# These wrap the C functions in src/gpu_launch.cpp which load PTX,
# extract device pointers from tinytorch tensors, and launch kernels.

#' Launch a compiled GPU kernel (elementwise)
#'
#' @param ptx Character, PTX assembly text.
#' @param kernel_name Character, kernel entry point name.
#' @param inputs List of torch_tensor (on CUDA device).
#' @param output torch_tensor, pre-allocated output (on CUDA device).
#' @param grid Integer vector of length 3, grid dimensions.
#' @param block Integer vector of length 3, block dimensions.
#' @param shared_mem Integer, shared memory bytes (default 0).
#' @return Invisible NULL. Output is written in-place.
#' @keywords internal
gpu_launch <- function(ptx, kernel_name, inputs, output,
                       grid, block, shared_mem = 0L) {
  C_gpu_launch(ptx, kernel_name, inputs, output,
        as.integer(grid), as.integer(block), as.integer(shared_mem))
  invisible(NULL)
}

#' Launch a compiled GPU reduction kernel
#'
#' @param ptx Character, PTX assembly text.
#' @param kernel_name Character, kernel entry point name.
#' @param input torch_tensor, single input (on CUDA device).
#' @param output torch_tensor, pre-allocated output (on CUDA device).
#' @param n_elements Integer, total elements per reduction.
#' @param grid Integer vector of length 3, grid dimensions.
#' @param block Integer vector of length 3, block dimensions.
#' @param shared_mem Integer, shared memory bytes.
#' @return Invisible NULL. Output is written in-place.
#' @keywords internal
gpu_launch_reduction <- function(ptx, kernel_name, input, output,
                                  n_elements, grid, block, shared_mem = 0L) {
  C_gpu_launch_reduction(ptx, kernel_name, input, output,
        as.integer(n_elements), as.integer(grid), as.integer(block),
        as.integer(shared_mem))
  invisible(NULL)
}

#' Launch a compiled GPU kernel (generic)
#'
#' Flexible launcher that takes a list of tensor arguments and a list of
#' scalar arguments. Args are packed in order: [tensor_ptrs..., scalars...].
#' Scalars can be integer (passed as i32) or numeric (passed as f32).
#'
#' @param ptx Character, PTX assembly text.
#' @param kernel_name Character, kernel entry point name.
#' @param tensors List of torch_tensor (on CUDA device).
#' @param scalars List of scalar values (integer or numeric).
#' @param grid Integer vector of length 3, grid dimensions.
#' @param block Integer vector of length 3, block dimensions.
#' @param shared_mem Integer, shared memory bytes (default 0).
#' @return Invisible NULL.
#' @keywords internal
gpu_launch_generic <- function(ptx, kernel_name, tensors, scalars,
                                grid, block, shared_mem = 0L) {
  C_gpu_launch_generic(ptx, kernel_name, tensors, scalars,
        as.integer(grid), as.integer(block), as.integer(shared_mem))
  invisible(NULL)
}

#' Clear the GPU kernel cache
#'
#' @return Number of cached kernels cleared.
#' @keywords internal
gpu_kernel_cache_clear <- function() {
  C_gpu_kernel_cache_clear()
}

#' GPU kernel cache statistics
#'
#' @return List with n_cached and kernel_names.
#' @keywords internal
gpu_kernel_cache_stats <- function() {
  C_gpu_kernel_cache_stats()
}
