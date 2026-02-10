.onLoad <- function(libname, pkgname) {
  # Ensure libtorch shared libraries are on the library path
  torch_home <- system.file(package = "torch")
  if (nzchar(torch_home)) {
    lib_path <- file.path(torch_home, "lib")
    # On Linux, add to LD_LIBRARY_PATH equivalent via dyn.load search
    if (.Platform$OS.type == "unix") {
      current <- Sys.getenv("LD_LIBRARY_PATH", "")
      if (!grepl(lib_path, current, fixed = TRUE)) {
        Sys.setenv(LD_LIBRARY_PATH = paste(lib_path, current, sep = ":"))
      }
    }
  }
}
