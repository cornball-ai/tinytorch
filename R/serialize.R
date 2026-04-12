# Minimal serialization helpers.
# torch_save / torch_load use RDS serialization for now.

#' Save a torch object to a file
#'
#' @param obj Object to save.
#' @param path File path.
#' @return Invisible NULL.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' x <- torch_randn(c(2, 3))
#' f <- tempfile(fileext = ".pt")
#' torch_save(x, f)
#' unlink(f)
#' }
#' }
torch_save <- function(obj, path) {
  saveRDS(obj, path)
  invisible(NULL)
}

#' Load a torch object from a file
#'
#' @param path File path.
#' @return The loaded object.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' x <- torch_randn(c(2, 3))
#' f <- tempfile(fileext = ".pt")
#' torch_save(x, f)
#' y <- torch_load(f)
#' unlink(f)
#' }
#' }
torch_load <- function(path) {
  readRDS(path)
}

#' Serialize a torch object to a raw vector
#'
#' @param obj Object to serialize.
#' @return Raw vector.
#' @export
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' bytes <- torch_serialize(torch_randn(c(2, 3)))
#' }
#' }
torch_serialize <- function(obj) {
  serialize(obj, NULL)
}

