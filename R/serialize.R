# Minimal serialization stubs
# torch_save/torch_load use RDS serialization for now.
# jit_trace is not yet supported.

#' Save a torch object to a file
#'
#' @param obj Object to save.
#' @param path File path.
#' @return Invisible NULL.
#' @export
torch_save <- function(obj, path) {
  saveRDS(obj, path)
  invisible(NULL)
}

#' Load a torch object from a file
#'
#' @param path File path.
#' @return The loaded object.
#' @export
torch_load <- function(path) {
  readRDS(path)
}

#' Serialize a torch object to a raw vector
#'
#' @param obj Object to serialize.
#' @return Raw vector.
#' @export
torch_serialize <- function(obj) {
  serialize(obj, NULL)
}

#' @keywords internal
jit_trace <- function(fn, ...) {
  stop("jit_trace is not yet supported in tinytorch. Use compile = FALSE.",
       call. = FALSE)
}
