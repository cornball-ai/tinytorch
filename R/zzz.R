#' @useDynLib tinytorch, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom utils capture.output download.file getFromNamespace object.size unzip
#' @importFrom stats setNames
NULL

.onLoad <- function(libname, pkgname) {
  # In stub mode (no libtorch), nothing to do
  if (!tryCatch(.Call("_tinytorch_C_rtorch_ping") == 1L, error = function(e) FALSE)) {
    return(invisible())
  }

  # Real backend: ensure libtorch shared libraries are findable.
  # The rpath baked into tinytorch.so at compile time handles this for most cases.
  # Add LIBTORCH_HOME/lib to LD_LIBRARY_PATH as a fallback.
  libtorch_home <- Sys.getenv("LIBTORCH_HOME", "")
  if (nzchar(libtorch_home)) {
    lib_path <- file.path(libtorch_home, "lib")
    current <- Sys.getenv("LD_LIBRARY_PATH", "")
    if (!grepl(lib_path, current, fixed = TRUE)) {
      Sys.setenv(LD_LIBRARY_PATH = paste(lib_path, current, sep = ":"))
    }
  }
}
