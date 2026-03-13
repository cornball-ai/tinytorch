if (!tinytorch::is_available()) exit_file("LibTorch not available")

# ===== compile() on nn_module =====

m <- nn_linear(10, 5)
x <- torch_randn(1, 10)

compiled <- compile(m, input = x)
expect_true(inherits(compiled, "compiled_module"),
            info = "compile(nn_linear) returns compiled_module")
expect_true(is.function(compiled$fn),
            info = "compiled_module has callable fn")
expect_true(length(compiled$params) > 0,
            info = "compiled_module has parameters")
expect_equal(compiled$meta$class, "nn_linear",
             info = "metadata class is nn_linear")
expect_equal(compiled$meta$forward_args, "input",
             info = "metadata forward_args is 'input'")

# Callable and correct
result <- compiled$fn(input = x)
expect_true(inherits(result, "torch_tensor"),
            info = "compiled fn returns a tensor")
expect_equal(as.integer(result$shape), c(1L, 5L),
             info = "compiled fn output shape is correct")

# Verify correctness against eager
eager_result <- m(x)
expect_true(torch_allclose(result, eager_result, atol = 1e-5),
            info = "compiled result matches eager execution")


# ===== compile() on plain function =====

cfn <- compile(function(x, y) x$matmul(y)$relu(),
               x = torch_randn(3, 3), y = torch_randn(3, 3))
expect_true(inherits(cfn, "compiled_module"),
            info = "compile(function) returns compiled_module")
expect_equal(cfn$meta$class, "function",
             info = "function compile has class 'function'")
expect_equal(sort(cfn$meta$forward_args), c("x", "y"),
             info = "function forward_args are x and y")

a <- torch_randn(3, 3)
b <- torch_randn(3, 3)
fn_result <- cfn$fn(x = a, y = b)
expect_true(inherits(fn_result, "torch_tensor"),
            info = "compiled function returns tensor")
expected <- a$matmul(b)$relu()
expect_true(torch_allclose(fn_result, expected, atol = 1e-5),
            info = "compiled function result is correct")


# ===== compile() with graph breaks =====

# Create a module with a side-effect call (causes graph break)
break_mod <- nn_module(
  "break_mod",
  initialize = function() {
    self$linear <- nn_linear(5, 3)
  },
  forward = function(input) {
    x <- self$linear(input)
    cat("debug\n")
    x$relu()
  }
)
bm <- break_mod()

# Should warn about graph breaks
expect_warning(
  compiled_break <- compile(bm, input = torch_randn(1, 5)),
  pattern = "graph break",
  info = "compile() warns on graph breaks"
)
expect_true(inherits(compiled_break, "compiled_module"),
            info = "compiled module with breaks is still a compiled_module")
expect_true(length(compiled_break$graph_breaks) > 0,
            info = "graph breaks are recorded")

# Still callable and correct via fallback
break_result <- compiled_break$fn(input = torch_randn(1, 5))
expect_true(inherits(break_result, "torch_tensor"),
            info = "module with graph breaks still executes")


# ===== compile() error cases =====

expect_error(compile(42, x = torch_randn(3)),
             pattern = "nn_module or a function",
             info = "compile() errors on non-module/function")

expect_error(compile(nn_linear(5, 3)),
             pattern = "named example input",
             info = "compile() errors without example inputs")

expect_error(compile(nn_linear(5, 3), torch_randn(1, 5)),
             pattern = "named",
             info = "compile() errors with unnamed inputs")


# ===== print.compiled_module =====

out <- capture.output(print(compiled))
expect_true(any(grepl("Compiled module", out)),
            info = "print shows 'Compiled module'")
expect_true(any(grepl("nn_linear", out)),
            info = "print shows class name")
expect_true(any(grepl("Parameters", out)),
            info = "print shows parameter count")
expect_true(any(grepl("Graph breaks", out)),
            info = "print shows graph break count")


# ===== save/load round-trip =====

artifact_path <- file.path(tempdir(), "test_model.torchlang")
unlink(artifact_path, recursive = TRUE)

compile(m, input = x, path = artifact_path)
expect_true(dir.exists(artifact_path),
            info = "save creates artifact directory")
expect_true(file.exists(file.path(artifact_path, "meta.rds")),
            info = "artifact has meta.rds")
expect_true(file.exists(file.path(artifact_path, "ir.rds")),
            info = "artifact has ir.rds")
expect_true(file.exists(file.path(artifact_path, "params.rds")),
            info = "artifact has params.rds")

loaded <- load_compiled(artifact_path)
expect_true(inherits(loaded, "compiled_module"),
            info = "load() returns compiled_module")
expect_equal(loaded$meta$class, "nn_linear",
             info = "loaded metadata class is correct")
expect_equal(length(loaded$params), length(compiled$params),
             info = "loaded has same number of params as compiled")

# Verify loaded result matches compiled result
loaded_result <- loaded$fn(input = x)
expect_true(inherits(loaded_result, "torch_tensor"),
            info = "loaded fn returns tensor")
expect_true(torch_allclose(loaded_result, eager_result, atol = 1e-5),
            info = "loaded result matches eager execution")


# ===== load() error cases =====

expect_error(load_compiled("/nonexistent/path.torchlang"),
             pattern = "not found",
             info = "load() errors on missing path")

bad_dir <- file.path(tempdir(), "bad.torchlang")
dir.create(bad_dir, recursive = TRUE, showWarnings = FALSE)
expect_error(load_compiled(bad_dir),
             pattern = "Missing meta.rds",
             info = "load() errors on missing meta.rds")
unlink(bad_dir, recursive = TRUE)


# ===== save path validation =====

expect_error(compile(m, input = x, path = "/tmp/bad_extension.xyz"),
             pattern = "\\.torchlang",
             info = "save requires .torchlang extension")


# ===== compile function save/load round-trip =====

fn_path <- file.path(tempdir(), "test_fn.torchlang")
unlink(fn_path, recursive = TRUE)

compile(function(x, y) x$matmul(y)$relu(),
        x = torch_randn(3, 3), y = torch_randn(3, 3),
        path = fn_path)

loaded_fn <- load_compiled(fn_path)
expect_true(inherits(loaded_fn, "compiled_module"),
            info = "loaded function artifact is compiled_module")
expect_equal(loaded_fn$meta$class, "function",
             info = "loaded function has class 'function'")

fn_a <- torch_randn(3, 3)
fn_b <- torch_randn(3, 3)
loaded_fn_result <- loaded_fn$fn(x = fn_a, y = fn_b)
expected_fn <- fn_a$matmul(fn_b)$relu()
expect_true(torch_allclose(loaded_fn_result, expected_fn, atol = 1e-5),
            info = "loaded function result is correct")

# Cleanup
unlink(artifact_path, recursive = TRUE)
unlink(fn_path, recursive = TRUE)
