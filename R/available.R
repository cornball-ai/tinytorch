#' Check if Rtorch backend is available
#'
#' Returns TRUE if the package was compiled with libtorch support.
#' When FALSE, all tensor operations will error. Use this to guard
#' code that requires the libtorch backend.
#'
#' @return Logical scalar.
#' @export
#' @examples
#' is_available()
is_available <- function() {
  tryCatch(.Call("_Rtorch_C_rtorch_ping") == 1L, error = function(e) FALSE)
}

#' Install libtorch
#'
#' Downloads and installs a pinned release of libtorch (PyTorch C++ distribution).
#' After installation, set LIBTORCH_HOME and reinstall Rtorch to enable the backend.
#'
#' @param path Installation directory. Default: ~/.local/lib/libtorch
#' @param version Pinned libtorch version. Updated deliberately with each release.
#' @return The installation path (invisibly).
#' @export
#' @examples
#' \dontrun{
#' rtorch_install_libtorch()
#' }
rtorch_install_libtorch <- function(path = "~/.local/lib/libtorch",
                                    version = "2.7.0") {
  path <- normalizePath(path, mustWork = FALSE)
  parent <- dirname(path)
  if (!dir.exists(parent)) dir.create(parent, recursive = TRUE)

  url <- sprintf(
    "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-%s%%2Bcpu.zip",
    version
  )

  tmp <- tempfile(fileext = ".zip")
  on.exit(unlink(tmp))

  message("Downloading libtorch ", version, " ...")
  download.file(url, tmp, mode = "wb")
  message("Extracting to ", parent, " ...")
  unzip(tmp, exdir = parent)

  message("LibTorch installed to: ", path)
  message("Reinstall Rtorch to enable the backend:")
  message("  Sys.setenv(LIBTORCH_HOME = \"", path, "\")")
  message("  install.packages(\"Rtorch\")")
  invisible(path)
}
