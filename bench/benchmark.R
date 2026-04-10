# 4-way benchmark: Python torch vs R torch vs Rcpp vs tinytorch
# All using libtorch 2.11.0. Saves results to bench/results.csv

timeit <- function(expr, n = 10000L, envir = parent.frame()) {
  expr <- substitute(expr)
  for (i in seq_len(100L)) eval(expr, envir)
  gc(FALSE)
  t0 <- proc.time()[[3L]]
  for (i in seq_len(n)) eval(expr, envir)
  elapsed <- proc.time()[[3L]] - t0
  elapsed / n * 1e6
}

timeit_subprocess <- function(setup, expr_str, n = 10000L) {
  script <- sprintf('
    timeit <- function(expr, n) {
      expr <- substitute(expr)
      for (i in seq_len(100L)) eval(expr, parent.frame())
      gc(FALSE)
      t0 <- proc.time()[[3L]]
      for (i in seq_len(n)) eval(expr, parent.frame())
      elapsed <- proc.time()[[3L]] - t0
      elapsed / n * 1e6
    }
    %s
    cat(timeit(%s, n = %dL))
  ', setup, expr_str, n)
  val <- system2("r", args = c("-e", shQuote(script)), stdout = TRUE, stderr = FALSE)
  as.numeric(val[length(val)])
}

# ---- Python results ----
cat("Running Python benchmark (torch 2.11.0)...\n")
py_json <- system2(
  "/tmp/torch-bench/bin/python",
  args = file.path(getwd(), "bench/benchmark_python.py"),
  stdout = TRUE, stderr = FALSE
)
py <- jsonlite::fromJSON(py_json[length(py_json)])

# ---- Rcpp setup ----
cat("Compiling Rcpp wrapper...\n")
torch_home <- system.file(package = "torch")
Sys.setenv(
  PKG_CXXFLAGS = paste0(
    "-std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1",
    " -I", torch_home, "/include",
    " -I", torch_home, "/include/torch/csrc/api/include"
  ),
  PKG_LIBS = paste0(
    "-L", torch_home, "/lib -ltorch -ltorch_cpu -lc10",
    " -Wl,-rpath,", torch_home, "/lib"
  )
)
Rcpp::sourceCpp("~/torch.cpp")

# ---- R setup ----
x_tiny  <- tinytorch::torch_randn(c(10, 10))
x_torch <- torch::torch_randn(c(10, 10))
x_rcpp  <- rcpp_torch_randn(c(10L, 10L))

# ---- Collect results ----
tests <- c(
  "namespace_add", "method_add", "chained_matmul",
  "method_chain", "creation", "large_matmul"
)

labels <- c(
  "Function add (10x10)", "Method .add (10x10)",
  "Chained matmul (10x10)", "Method chain .add().mul() (10x10)",
  "Creation randn(10,10)", "Large matmul (1000x1000)"
)

cat("Running R benchmarks...\n")

r_torch <- r_rcpp <- r_tiny <- numeric(6)

# 1: namespace add
r_torch[1] <- timeit(torch::torch_add(x_torch, x_torch))
r_rcpp[1]  <- timeit(rcpp_torch_add(x_rcpp, x_rcpp))
r_tiny[1]  <- timeit(tinytorch::torch_add(x_tiny, x_tiny))

# 2: method $add (subprocess)
r_torch[2] <- timeit_subprocess("x <- torch::torch_randn(c(10, 10))", "x$add(x)")
r_rcpp[2]  <- NA  # no $ dispatch
r_tiny[2]  <- timeit_subprocess("x <- tinytorch::torch_randn(c(10, 10))", "x$add(x)")

# 3: chained matmul
r_torch[3] <- timeit({
  tmp <- torch::torch_matmul(x_torch, x_torch)
  torch::torch_matmul(tmp, x_torch)
})
r_rcpp[3] <- timeit({
  tmp <- rcpp_torch_matmul(x_rcpp, x_rcpp)
  rcpp_torch_matmul(tmp, x_rcpp)
})
r_tiny[3] <- timeit({
  tmp <- tinytorch::torch_matmul(x_tiny, x_tiny)
  tinytorch::torch_matmul(tmp, x_tiny)
})

# 4: method chain (subprocess)
r_torch[4] <- timeit_subprocess("x <- torch::torch_randn(c(10, 10))", "x$add(x)$mul(x)")
r_rcpp[4]  <- NA
r_tiny[4]  <- timeit_subprocess("x <- tinytorch::torch_randn(c(10, 10))", "x$add(x)$mul(x)")

# 5: creation
r_torch[5] <- timeit(torch::torch_randn(c(10, 10)))
r_rcpp[5]  <- timeit(rcpp_torch_randn(c(10L, 10L)))
r_tiny[5]  <- timeit(tinytorch::torch_randn(c(10, 10)))

# 6: large matmul
big_torch <- torch::torch_randn(c(1000, 1000))
big_tiny  <- tinytorch::torch_randn(c(1000, 1000))
big_rcpp  <- rcpp_torch_randn(c(1000L, 1000L))
r_torch[6] <- timeit(torch::torch_matmul(big_torch, big_torch), n = 200L)
r_rcpp[6]  <- timeit(rcpp_torch_matmul(big_rcpp, big_rcpp), n = 200L)
r_tiny[6]  <- timeit(tinytorch::torch_matmul(big_tiny, big_tiny), n = 200L)

# ---- Build matrix ----
py_vals <- vapply(tests, function(t) py[[t]], numeric(1))

mat <- data.frame(
  test      = labels,
  python_us = round(py_vals, 1),
  torch_us  = round(r_torch, 1),
  rcpp_us   = round(r_rcpp, 1),
  tiny_us   = round(r_tiny, 1),
  stringsAsFactors = FALSE
)

outfile <- file.path(getwd(), "bench/results.csv")
write.csv(mat, outfile, row.names = FALSE)

cat("\n=== Results (microseconds per call, libtorch 2.11.0) ===\n\n")
print(mat, right = FALSE, row.names = FALSE)
cat(sprintf("\nSaved to %s\n", outfile))
