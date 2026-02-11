#' Compiled Graph Cache
#'
#' Caches traced functions to avoid re-compilation overhead.
#' Cache key includes expression structure and tensor shapes.

# Global cache environment
.torchlang_cache <- new.env(parent = emptyenv())

#' Generate Cache Key
#'
#' Creates a cache key from expression and tensor shapes.
#'
#' @param expr_key Character, deparsed expression
#' @param tensor_shapes List of tensor shape vectors
#' @return Character cache key
#' @noRd
make_cache_key <- function(expr_key, tensor_shapes) {
  shapes_str <- paste(
    vapply(tensor_shapes, function(s) paste(s, collapse = "x"), character(1)),
    collapse = "_"
  )
  paste0(expr_key, "|", shapes_str)
}


#' Get Cached Traced Function
#'
#' @param key Cache key
#' @return Traced function or NULL if not cached
#' @noRd
cache_get <- function(key) {
  if (exists(key, envir = .torchlang_cache, inherits = FALSE)) {
    get(key, envir = .torchlang_cache, inherits = FALSE)
  } else {
    NULL
  }
}


#' Store Traced Function in Cache
#'
#' @param key Cache key
#' @param traced_fn Traced function
#' @noRd
cache_set <- function(key, traced_fn) {
  assign(key, traced_fn, envir = .torchlang_cache)
}


#' Clear the Compilation Cache
#'
#' Removes all cached traced functions.
#'
#' @return Invisibly returns the number of items cleared
#' @examples
#' clear_cache()
#' @export
clear_cache <- function() {
  n <- length(ls(.torchlang_cache))
  rm(list = ls(.torchlang_cache), envir = .torchlang_cache)
  invisible(n)
}


#' Get Cache Statistics
#'
#' @return List with cache size and keys
#' @examples
#' cache_stats()
#' @export
cache_stats <- function() {
  keys <- ls(.torchlang_cache)
  list(
    size = length(keys),
    keys = keys
  )
}
