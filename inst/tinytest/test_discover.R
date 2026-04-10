if (!tinytorch::is_available()) exit_file("LibTorch not available")

# ===== Module Discovery =====

# Simple model for testing
simple_model <- nn_module(
  initialize = function() {
    self$norm <- nn_layer_norm(32)
    self$fc1 <- nn_linear(32, 128)
    self$fc2 <- nn_linear(128, 32)
  },
  forward = function(x) {
    x <- self$norm(x)
    x <- nnf_relu(self$fc1(x))
    self$fc2(x)
  }
)()

# ===== discover_modules =====

tree <- discover_modules(simple_model)

expect_true(inherits(tree, "module_tree"),
            info = "discover_modules returns module_tree")

# Root + 3 sub-modules = 4
expect_equal(length(tree), 4L,
             info = "Simple model has 4 modules (root + 3 sub)")

# Check root
root <- tree[[1]]
expect_equal(root$path, "",
             info = "Root has empty path")
expect_true("x" %in% root$forward_args,
            info = "Root forward takes x")

# Check sub-modules found
paths <- vapply(tree, function(i) i$path, character(1))
expect_true("norm" %in% paths,
            info = "Finds norm sub-module")
expect_true("fc1" %in% paths,
            info = "Finds fc1 sub-module")
expect_true("fc2" %in% paths,
            info = "Finds fc2 sub-module")

# Check parameter counts
norm_info <- tree[[which(paths == "norm")]]
expect_equal(norm_info$n_params, 2L,
             info = "LayerNorm has 2 params (weight + bias)")

fc1_info <- tree[[which(paths == "fc1")]]
expect_equal(fc1_info$n_params, 2L,
             info = "Linear has 2 params (weight + bias)")

# ===== Nested model =====

nested_model <- nn_module(
  initialize = function() {
    self$block <- nn_module(
      initialize = function() {
        self$ln <- nn_layer_norm(16)
        self$fc <- nn_linear(16, 16)
      },
      forward = function(x) {
        self$fc(self$ln(x))
      }
    )()
    self$out <- nn_linear(16, 4)
  },
  forward = function(x) {
    self$out(self$block(x))
  }
)()

tree2 <- discover_modules(nested_model)
paths2 <- vapply(tree2, function(i) i$path, character(1))

expect_true("block.ln" %in% paths2,
            info = "Finds nested sub-modules (block.ln)")
expect_true("block.fc" %in% paths2,
            info = "Finds nested sub-modules (block.fc)")
expect_true("out" %in% paths2,
            info = "Finds top-level sub-modules (out)")

# ===== find_torch_packages =====

pkgs <- find_torch_packages()
expect_true(is.character(pkgs),
            info = "find_torch_packages returns character vector")
# tinytorch itself depends on torch
expect_true("tinytorch" %in% pkgs,
            info = "tinytorch found as torch-dependent package")

# ===== find_modules_in_package (installed package) =====

# Test on tinytorch itself (no nn_modules in R/)
tl_mods <- find_modules_in_package("tinytorch")
expect_true(is.data.frame(tl_mods),
            info = "find_modules_in_package returns data.frame")

# ===== Error handling =====

expect_error(discover_modules("not_a_module"),
             info = "discover_modules errors on non-module input")

expect_error(find_modules_in_package("nonexistent_package_xyz"),
             info = "find_modules_in_package errors on missing package")
