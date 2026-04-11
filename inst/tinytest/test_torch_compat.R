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
     cat(paste(exports, collapse = "\\n"))'
  )), stdout = TRUE, stderr = FALSE)
  if (length(out) == 0) NULL else out
}, error = function(e) NULL)

if (is.null(torch_exports) || length(torch_exports) < 10) {
  exit_file("torch package not available or not functional")
}

tiny_exports <- getNamespaceExports("tinytorch")

# ---- Core ops (must have) ----

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
  "torch_dropout", "torch_layer_norm", "torch_batch_norm",
  "torch_flatten", "torch_squeeze", "torch_unsqueeze",
  "torch_transpose", "torch_clone", "torch_neg"
)

for (op in core_ops) {
  expect_true(op %in% tiny_exports,
              info = sprintf("Core op '%s' missing from tinytorch", op))
}

# ---- Core modules (must have) ----

core_modules <- c(
  "nn_module", "nn_linear", "nn_relu", "nn_gelu", "nn_sequential",
  "nn_layer_norm", "nn_embedding", "nn_conv1d", "nn_conv2d",
  "nn_batch_norm1d", "nn_batch_norm2d", "nn_dropout",
  "nn_lstm", "nn_gru", "nn_rnn"
)

for (mod in core_modules) {
  expect_true(mod %in% tiny_exports,
              info = sprintf("Core module '%s' missing from tinytorch", mod))
}

# ---- Core functional ops (must have) ----

core_nnf <- c(
  "nnf_relu", "nnf_gelu", "nnf_silu", "nnf_softmax", "nnf_log_softmax",
  "nnf_layer_norm", "nnf_linear", "nnf_conv1d", "nnf_conv2d",
  "nnf_dropout", "nnf_embedding", "nnf_cross_entropy",
  "nnf_mse_loss", "nnf_binary_cross_entropy"
)

for (fn in core_nnf) {
  expect_true(fn %in% tiny_exports,
              info = sprintf("Core nnf '%s' missing from tinytorch", fn))
}

# ---- Core optimizers (must have) ----

core_optim <- c("optim_sgd", "optim_adam", "optim_adamw", "optim_rmsprop", "optim_adagrad")

for (opt in core_optim) {
  expect_true(opt %in% tiny_exports,
              info = sprintf("Core optimizer '%s' missing from tinytorch", opt))
}

# ---- Core distributions (must have) ----

core_distr <- c("distr_normal", "distr_bernoulli", "distr_categorical")

for (d in core_distr) {
  expect_true(d %in% tiny_exports,
              info = sprintf("Core distribution '%s' missing from tinytorch", d))
}

# ---- Core data loading (must have) ----

core_data <- c("dataset", "dataloader", "tensor_dataset")

for (d in core_data) {
  expect_true(d %in% tiny_exports,
              info = sprintf("Core data util '%s' missing from tinytorch", d))
}

# ---- Dtype constants (must have) ----

core_dtypes <- c(
  "torch_float32", "torch_float64", "torch_float16", "torch_bfloat16",
  "torch_int32", "torch_int64", "torch_int8", "torch_uint8", "torch_bool",
  "torch_float8_e4m3fn", "torch_float8_e5m2"
)

for (dt in core_dtypes) {
  expect_true(dt %in% tiny_exports,
              info = sprintf("Core dtype '%s' missing from tinytorch", dt))
}

# ---- Overall coverage ----

# Known intentional gaps (not counted as missing)
intentional_gaps <- c(
  # JIT/TorchScript (separate project)
  grep("^jit_", torch_exports, value = TRUE),
  # Ignite optimizers (torch-specific contrib)
  grep("^optim_ignite|^optimizer_ignite|^OptimizerIgnite", torch_exports, value = TRUE),
  # CUDA debug/profiling tools
  "cuda_amp_grad_scaler", "cuda_dump_memory_snapshot",
  "cuda_memory_snapshot", "cuda_memory_summary", "cuda_record_memory_history",
  # Internal torch plumbing
  "call_torch_function", "buffer_from_torch_tensor",
  "get_install_libs_url", "install_torch_from_file",
  # Pipe operator (users have their own)
  "%>%",
  # Contrib extensions
  "contrib_sort_vertices"
)

# Coverage excluding intentional gaps
torch_relevant <- setdiff(torch_exports, intentional_gaps)
covered <- sum(torch_relevant %in% tiny_exports)
total <- length(torch_relevant)
pct <- round(100 * covered / total, 1)
message(sprintf("torch:: coverage (excluding intentional gaps): %d/%d (%.1f%%)", covered, total, pct))

# Must be above 95%
expect_true(pct >= 95,
            info = sprintf("Coverage dropped below 95%%: %.1f%% (%d/%d)", pct, covered, total))

# Report any non-intentional gaps
non_intentional_missing <- setdiff(torch_relevant, tiny_exports)
if (length(non_intentional_missing) > 0) {
  message("Non-intentional gaps: ", paste(non_intentional_missing, collapse = ", "))
}

# ---- Argument compatibility (first-arg check) ----

torch_torch_fns <- torch_exports[grepl("^torch_", torch_exports)]
shared <- intersect(torch_torch_fns, tiny_exports)
if (length(shared) > 0) {
  tiny_formals <- lapply(shared, function(fn) {
    f <- get(fn, envir = asNamespace("tinytorch"))
    if (is.function(f)) names(formals(f)) else NULL
  })
  names(tiny_formals) <- shared

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

  for (line in torch_formals_raw) {
    parts <- strsplit(line, ":", fixed = TRUE)[[1]]
    if (length(parts) != 2) next
    fn <- parts[1]
    torch_args <- strsplit(parts[2], ",")[[1]]
    tiny_args <- tiny_formals[[fn]]
    if (is.null(tiny_args) || length(torch_args) == 0) next
    expect_equal(tiny_args[1], torch_args[1],
                 info = sprintf("%s: first arg '%s' vs torch '%s'",
                                fn, tiny_args[1], torch_args[1]))
  }
}
