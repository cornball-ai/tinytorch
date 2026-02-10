# Minimal serialization stubs
# torch_save/torch_load use RDS serialization for now.
# jit_trace is not yet supported (compile.R falls back gracefully).

torch_save <- function(obj, path) {
  saveRDS(obj, path)
}

torch_load <- function(path) {
  readRDS(path)
}

jit_trace <- function(fn, ...) {
  stop("jit_trace is not yet supported in Rtorch. Use compile = FALSE.",
       call. = FALSE)
}
