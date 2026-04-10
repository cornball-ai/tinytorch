# Test API compatibility with torch (mlverse)
# Verifies tinytorch exports matching function names and argument signatures.
# Only runs locally (needs torch installed, can't coexist in same session).

if (!at_home()) exit_file("torch compatibility test only runs locally")

library(tinytorch)

# Run torch export discovery in a subprocess (can't load both in same session)
torch_exports <- tryCatch({
  out <- system2("r", args = c("-e", shQuote(
    'if (!requireNamespace("torch", quietly = TRUE)) quit("no", 1L);
     exports <- getNamespaceExports("torch");
     torch_fns <- exports[grepl("^torch_", exports)];
     nn_fns <- exports[grepl("^nn_", exports)];
     nnf_fns <- exports[grepl("^nnf_", exports)];
     cat(paste(c(torch_fns, nn_fns, nnf_fns), collapse = "\\n"))'
  )), stdout = TRUE, stderr = FALSE)
  if (length(out) == 0) NULL else out
}, error = function(e) NULL)

if (is.null(torch_exports) || length(torch_exports) < 10) {
  exit_file("torch package not available or not functional")
}

tiny_exports <- getNamespaceExports("tinytorch")

# Core torch_* functions that tinytorch MUST have
core_ops <- c(
  "torch_tensor", "torch_zeros", "torch_ones", "torch_randn", "torch_empty",
  "torch_arange", "torch_linspace", "torch_full",
  "torch_add", "torch_sub", "torch_mul", "torch_div",
  "torch_matmul", "torch_mm", "torch_bmm",
  "torch_sum", "torch_mean", "torch_max", "torch_min",
  "torch_exp", "torch_log", "torch_sqrt", "torch_abs",
  "torch_relu", "torch_sigmoid", "torch_tanh",
  "torch_cat", "torch_stack", "torch_reshape",
  "torch_softmax", "torch_log_softmax",
  "torch_conv1d", "torch_conv2d",
  "torch_linear", "torch_embedding",
  "torch_dropout", "torch_layer_norm", "torch_batch_norm"
)

for (op in core_ops) {
  expect_true(op %in% tiny_exports,
              info = sprintf("Core op '%s' missing from tinytorch", op))
}

# nn_* modules that tinytorch MUST have
core_modules <- c(
  "nn_module", "nn_linear", "nn_relu", "nn_gelu", "nn_sequential",
  "nn_layer_norm", "nn_embedding"
)

for (mod in core_modules) {
  expect_true(mod %in% tiny_exports,
              info = sprintf("Core module '%s' missing from tinytorch", mod))
}

# Report coverage of torch's torch_* exports
torch_torch_fns <- torch_exports[grepl("^torch_", torch_exports)]
covered <- sum(torch_torch_fns %in% tiny_exports)
total <- length(torch_torch_fns)
pct <- round(100 * covered / total, 1)
message(sprintf("torch_* coverage: %d/%d (%.1f%%)", covered, total, pct))

# Check argument compatibility for shared functions
shared <- intersect(torch_torch_fns, tiny_exports)
if (length(shared) > 0) {
  # Get tinytorch formals for shared ops
  tiny_formals <- lapply(shared, function(fn) {
    f <- get(fn, envir = asNamespace("tinytorch"))
    if (is.function(f)) names(formals(f)) else NULL
  })
  names(tiny_formals) <- shared

  # Get torch formals via subprocess
  torch_formals_raw <- system2("r", args = c("-e", shQuote(sprintf(
    'fns <- c(%s);
     for (fn in fns) {
       f <- tryCatch(get(fn, envir = asNamespace("torch")), error = function(e) NULL);
       if (is.function(f)) {
         args <- paste(names(formals(f)), collapse = ",");
         cat(fn, ":", args, "\\n", sep = "")
       }
     }',
    paste(shQuote(shared[1:min(50, length(shared))]), collapse = ",")
  ))), stdout = TRUE, stderr = FALSE)

  # Parse and compare first args (self, input, etc.)
  for (line in torch_formals_raw) {
    parts <- strsplit(line, ":", fixed = TRUE)[[1]]
    if (length(parts) != 2) next
    fn <- parts[1]
    torch_args <- strsplit(parts[2], ",")[[1]]
    tiny_args <- tiny_formals[[fn]]
    if (is.null(tiny_args) || length(torch_args) == 0) next

    # First arg name should match (self, input, etc.)
    expect_equal(tiny_args[1], torch_args[1],
                 info = sprintf("%s: first arg '%s' vs torch '%s'",
                                fn, tiny_args[1], torch_args[1]))
  }
}
