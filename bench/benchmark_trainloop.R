# Training loop benchmark: tinytorch vs torch
#
# Reproduces the setup from mlverse/torch#268 and #970.
# Measures a single training step (forward + backward + optimizer step)
# for a small MLP, then scales hidden size to check for GC-induced
# step functions (the #970 phenomenon).
#
# Usage: r bench/benchmark_trainloop.R

timeit <- function(expr, n = 1000L, warmup = 100L, envir = parent.frame()) {
  expr <- substitute(expr)
  for (i in seq_len(warmup)) eval(expr, envir)
  gc(FALSE)
  t0 <- proc.time()[[3L]]
  for (i in seq_len(n)) eval(expr, envir)
  elapsed <- proc.time()[[3L]] - t0
  elapsed / n * 1e6
}

# Run a single benchmark in a subprocess to avoid S3/namespace conflicts
bench_subprocess <- function(pkg, hidden, batch = 32L, input_dim = 20L, n = 1000L) {
  if (pkg == "tinytorch") {
    script <- sprintf('
      library(tinytorch)
      h <- %dL; b <- %dL; d <- %dL

      model <- nn_sequential(
        nn_linear(d, h), nn_relu(),
        nn_linear(h, h), nn_relu(),
        nn_linear(h, 1L)
      )
      opt <- optim_adam(model$parameters(), lr = 0.001)
      X <- torch_randn(b, d)
      Y <- torch_randn(b, 1L)

      train_step <- function() {
        opt$zero_grad()
        pred <- model(X)
        loss <- nnf_mse_loss(pred, Y, 1L)
        loss$backward()
        opt$step()
      }

      for (i in seq_len(100L)) train_step()
      gc(FALSE)
      t0 <- proc.time()[[3L]]
      for (i in seq_len(%dL)) train_step()
      elapsed <- proc.time()[[3L]] - t0
      cat(elapsed / %dL * 1e6)
    ', hidden, batch, input_dim, n, n)
  } else {
    script <- sprintf('
      library(torch)
      h <- %dL; b <- %dL; d <- %dL

      model <- nn_sequential(
        nn_linear(d, h), nn_relu(),
        nn_linear(h, h), nn_relu(),
        nn_linear(h, 1L)
      )
      opt <- optim_adam(model$parameters, lr = 0.001)
      X <- torch_randn(b, d)
      Y <- torch_randn(b, 1L)

      train_step <- function() {
        opt$zero_grad()
        pred <- model(X)
        loss <- nnf_mse_loss(pred, Y)
        loss$backward()
        opt$step()
      }

      for (i in seq_len(100L)) train_step()
      gc(FALSE)
      t0 <- proc.time()[[3L]]
      for (i in seq_len(%dL)) train_step()
      elapsed <- proc.time()[[3L]] - t0
      cat(elapsed / %dL * 1e6)
    ', hidden, batch, input_dim, n, n)
  }
  val <- system2("r", args = c("-e", shQuote(script)),
                 stdout = TRUE, stderr = FALSE)
  as.numeric(val[length(val)])
}

# ---- 1. Single training step at fixed size ----
cat("=== Training step benchmark (hidden=64, batch=32, input=20) ===\n")
tiny_step <- bench_subprocess("tinytorch", 64L)
torch_step <- bench_subprocess("torch", 64L)
cat(sprintf("  tinytorch: %.0f us/step\n", tiny_step))
cat(sprintf("  torch:     %.0f us/step\n", torch_step))
cat(sprintf("  speedup:   %.1fx\n\n", torch_step / tiny_step))

# ---- 2. Scale hidden size (the #970 sweep) ----
cat("=== Scaling hidden size (batch=32, input=20) ===\n")
hidden_sizes <- seq(20L, 200L, by = 20L)
tiny_vals <- torch_vals <- numeric(length(hidden_sizes))

for (i in seq_along(hidden_sizes)) {
  h <- hidden_sizes[i]
  cat(sprintf("  h=%d ...", h))
  tiny_vals[i] <- bench_subprocess("tinytorch", h, n = 500L)
  torch_vals[i] <- bench_subprocess("torch", h, n = 500L)
  cat(sprintf(" tiny=%.0f us, torch=%.0f us (%.1fx)\n",
              tiny_vals[i], torch_vals[i], torch_vals[i] / tiny_vals[i]))
}

# ---- 3. Dataloader iteration overhead ----
cat("\n=== Dataloader iteration (1000 samples, batch=64) ===\n")
dl_tiny <- system2("r", args = c("-e", shQuote('
  library(tinytorch)
  ds <- tensor_dataset(torch_randn(c(1000L, 20L)), torch_randn(c(1000L, 1L)))
  dl <- dataloader(ds, batch_size = 64L)
  # Warmup
  for (j in 1:10) {
    it <- dataloader_make_iter(dl)
    repeat { b <- it$.next(); if (is.null(b)) break }
  }
  gc(FALSE)
  t0 <- proc.time()[[3L]]
  for (j in 1:100) {
    it <- dataloader_make_iter(dl)
    repeat { b <- it$.next(); if (is.null(b)) break }
  }
  elapsed <- proc.time()[[3L]] - t0
  cat(elapsed / 100 * 1e6)
')), stdout = TRUE, stderr = FALSE)
dl_tiny_us <- as.numeric(dl_tiny[length(dl_tiny)])

dl_torch <- system2("r", args = c("-e", shQuote('
  library(torch)
  ds <- tensor_dataset(torch_randn(c(1000L, 20L)), torch_randn(c(1000L, 1L)))
  dl <- dataloader(ds, batch_size = 64L)
  # Warmup
  for (j in 1:10) {
    coro::loop(for (b in dl) { })
  }
  gc(FALSE)
  t0 <- proc.time()[[3L]]
  for (j in 1:100) {
    coro::loop(for (b in dl) { })
  }
  elapsed <- proc.time()[[3L]] - t0
  cat(elapsed / 100 * 1e6)
')), stdout = TRUE, stderr = FALSE)
dl_torch_us <- as.numeric(dl_torch[length(dl_torch)])

cat(sprintf("  tinytorch: %.0f us/epoch\n", dl_tiny_us))
cat(sprintf("  torch:     %.0f us/epoch\n", dl_torch_us))
cat(sprintf("  speedup:   %.1fx\n", dl_torch_us / dl_tiny_us))

# ---- Save results ----
results <- data.frame(
  hidden = hidden_sizes,
  tinytorch_us = round(tiny_vals),
  torch_us = round(torch_vals),
  speedup = round(torch_vals / tiny_vals, 1)
)
outfile <- file.path("bench", "trainloop_results.csv")
write.csv(results, outfile, row.names = FALSE)
cat(sprintf("\nScaling results saved to %s\n", outfile))
