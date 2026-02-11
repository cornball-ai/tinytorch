# GPU Kernel Cache
#
# Caches compiled PTX kernels to avoid recompiling identical graph structures.
# Cache key: {ir_hash}_{sm_version}

.gpu_kernel_cache <- new.env(parent = emptyenv())

#' Get cached GPU kernel
#'
#' @param ir_hash Character hash of IR graph structure
#' @param sm_version Integer CUDA compute capability (e.g., 86 for sm_86)
#' @return Cached PTX string or NULL if not found
#' @examples
#' gpu_kernel_get("abc123", 86)
#' @export
gpu_kernel_get <- function(ir_hash, sm_version) {
  cache_key <- paste0(ir_hash, "_", sm_version)
  .gpu_kernel_cache[[cache_key]]
}

#' Set cached GPU kernel
#'
#' @param ir_hash Character hash of IR graph structure
#' @param sm_version Integer CUDA compute capability
#' @param ptx Character PTX assembly code
#' @examples
#' gpu_kernel_set("abc123", 86, "// ptx code")
#' @export
gpu_kernel_set <- function(ir_hash, sm_version, ptx) {
  cache_key <- paste0(ir_hash, "_", sm_version)
  .gpu_kernel_cache[[cache_key]] <- ptx
  invisible(ptx)
}

#' Clear GPU kernel cache
#'
#' @examples
#' gpu_kernel_clear()
#' @export
gpu_kernel_clear <- function() {
  rm(list = ls(.gpu_kernel_cache), envir = .gpu_kernel_cache)
  invisible(NULL)
}

#' Get cache statistics
#'
#' @return Named list with cache stats
#' @examples
#' gpu_kernel_stats()
#' @export
gpu_kernel_stats <- function() {
  list(
    n_kernels = length(ls(.gpu_kernel_cache)),
    total_size_bytes = sum(vapply(ls(.gpu_kernel_cache), function(k) {
      object.size(.gpu_kernel_cache[[k]])
    }, numeric(1)))
  )
}

#' Compute IR hash for cache key
#'
#' @param graph IR graph object
#' @return Character hash string
#' @examples
#' \donttest{
#' stmts <- list(quote(y <- x$relu()))
#' e <- new.env(); e$x <- torch_randn(c(2, 3))
#' g <- lower_to_ir(stmts, e)
#' compute_ir_hash(g)
#' }
#' @export
compute_ir_hash <- function(graph) {
  # Build structural fingerprint: node ops + connections
  parts <- character(length(graph$nodes))

  for (i in seq_along(graph$nodes)) {
    node <- graph$nodes[[i]]
    id <- names(graph$nodes)[i]

    # Format: id:op:inputs
    input_str <- paste(node$inputs, collapse = ",")
    parts[i] <- paste0(id, ":", node$op, ":", input_str)
  }

  # Hash the concatenated structure
  structure_str <- paste(parts, collapse = "|")
  digest::digest(structure_str, algo = "xxhash64")
}
