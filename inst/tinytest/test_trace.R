# ===== Helper utilities =====

is_nn_module_callable <- Rtorch:::is_nn_module_callable
get_module_instance <- Rtorch:::get_module_instance
resolve_self_access <- Rtorch:::resolve_self_access
get_private <- Rtorch:::get_private
expand_nnf <- Rtorch:::expand_nnf
is_sequential_body <- Rtorch:::is_sequential_body

# ===== Step 1: Helper functions =====

# is_nn_module_callable
expect_true(is_nn_module_callable(nn_linear(5, 3)),
            info = "nn_linear is an nn_module callable")
expect_true(is_nn_module_callable(nn_relu()),
            info = "nn_relu is an nn_module callable")
expect_false(is_nn_module_callable(function(x) x),
             info = "plain function is not nn_module")
expect_false(is_nn_module_callable(42),
             info = "numeric is not nn_module")

# get_module_instance
m <- nn_linear(10, 5)
inst <- get_module_instance(m)
expect_true(!is.null(inst),
            info = "get_module_instance returns non-NULL for nn_linear")
expect_true(is.environment(inst),
            info = "instance is an environment (module self)")

# resolve_self_access — parameters
resolved <- resolve_self_access("weight", inst, prefix = "")
expect_equal(resolved$type, "parameter",
             info = "self$weight resolves as parameter")
expect_equal(resolved$name, "weight",
             info = "parameter name is 'weight' with empty prefix")
expect_true(inherits(resolved$value, "torch_tensor"),
            info = "parameter value is a tensor")

resolved2 <- resolve_self_access("weight", inst, prefix = "layer1")
expect_equal(resolved2$name, "layer1.weight",
             info = "parameter name with prefix is 'layer1.weight'")

# resolve_self_access — scalars
resolved_s <- resolve_self_access("in_features", inst, prefix = "")
expect_equal(resolved_s$type, "scalar",
             info = "self$in_features resolves as scalar")

# resolve_self_access — sub-modules
seq_m <- nn_sequential(nn_linear(10, 20), nn_linear(20, 5))
seq_inst <- get_module_instance(seq_m)
resolved_mod <- resolve_self_access("0", seq_inst, prefix = "")
# nn_sequential stores sub-modules differently — they may be in modules_
# Check if we get module or scalar
# (nn_sequential sub-modules are in private$modules_)
priv <- get_private(seq_inst)
expect_true(length(priv$modules_) > 0,
            info = "nn_sequential has sub-modules in private$modules_")

# ===== Step 2: expand_nnf =====

# nnf_relu -> torch_relu
result <- expand_nnf("nnf_relu", list(quote(x)))
expect_true(!is.null(result), info = "nnf_relu expands")
expect_true(is.call(result), info = "expanded nnf_relu is a call")
expect_equal(as.character(result[[1]]), "torch_relu",
             info = "nnf_relu expands to torch_relu")

# nnf_linear -> torch_linear
result2 <- expand_nnf("nnf_linear",
                       list(quote(input), quote(weight), quote(bias)))
expect_true(!is.null(result2), info = "nnf_linear expands")
expect_equal(as.character(result2[[1]]), "torch_linear",
             info = "nnf_linear expands to torch_linear")

# nnf_gelu -> torch_gelu
result3 <- expand_nnf("nnf_gelu", list(quote(x), "none"))
expect_true(!is.null(result3), info = "nnf_gelu expands")

# nnf_dropout with training=FALSE -> identity
result4 <- expand_nnf("nnf_dropout", list(quote(x), 0.5, FALSE))
expect_true(is.symbol(result4) && as.character(result4) == "x",
            info = "nnf_dropout(training=FALSE) returns input unchanged")

# Unknown nnf_ -> NULL
result5 <- expand_nnf("nnf_nonexistent", list(quote(x)))
expect_true(is.null(result5), info = "unknown nnf_ returns NULL")

# ===== Step 3: expand_module — nn_linear =====

m_linear <- nn_linear(10, 5)
expanded <- expand_module(m_linear,
                          arg_exprs = list(input = as.symbol("x")))

expect_true(length(expanded$statements) > 0,
            info = "nn_linear expands to statements")
expect_true("weight" %in% names(expanded$params),
            info = "nn_linear expansion captures weight parameter")
expect_true("bias" %in% names(expanded$params),
            info = "nn_linear expansion captures bias parameter")
expect_equal(length(expanded$graph_breaks), 0,
             info = "nn_linear expands without graph breaks")

# The expanded statement should be a torch_linear call
stmt <- expanded$statements[[1]]
expect_true(is.call(stmt), info = "expanded nn_linear statement is a call")
if (is.symbol(stmt[[1]])) {
  fn_name <- as.character(stmt[[1]])
  expect_equal(fn_name, "torch_linear",
               info = "nn_linear expands to torch_linear")
}

# ===== Step 3b: expand_module — 2-layer MLP =====

mlp <- nn_sequential(nn_linear(10, 20), nn_relu(), nn_linear(20, 5))
expanded_mlp <- expand_module(mlp,
                               arg_exprs = list(input = as.symbol("x")))

expect_true(length(expanded_mlp$statements) > 0,
            info = "MLP expands to statements")
expect_true(length(expanded_mlp$params) > 0,
            info = "MLP captures parameters")
# Should have weight and bias for both linear layers
param_names <- names(expanded_mlp$params)
expect_true("0.weight" %in% param_names,
            info = "MLP has 0.weight parameter")
expect_true("0.bias" %in% param_names,
            info = "MLP has 0.bias parameter")
expect_true("2.weight" %in% param_names,
            info = "MLP has 2.weight parameter")
expect_true("2.bias" %in% param_names,
            info = "MLP has 2.bias parameter")

# ===== Step 4: Static if/else resolution =====

# nn_relu with inplace=FALSE resolves the if branch
relu_m <- nn_relu()
expanded_relu <- expand_module(relu_m,
                                arg_exprs = list(input = as.symbol("x")))
expect_true(length(expanded_relu$statements) > 0,
            info = "nn_relu expands")
expect_equal(length(expanded_relu$graph_breaks), 0,
             info = "nn_relu expands without graph breaks")

# ===== Step 5: nn_sequential unrolling =====

seq3 <- nn_sequential(
  nn_linear(10, 20),
  nn_relu(),
  nn_linear(20, 5)
)
expanded_seq <- expand_module(seq3,
                               arg_exprs = list(input = as.symbol("x")))

# Should produce assignment statements for each layer
expect_true(length(expanded_seq$statements) >= 3,
            info = "sequential unrolls to multiple statements")

# ===== Step 6: Graph break detection =====

# A module with x$size() should produce graph breaks
custom_mod <- nn_module(
  initialize = function(d) {
    self$linear <- nn_linear(d, d)
  },
  forward = function(x) {
    n <- x$size(1)  # data-dependent
    self$linear(x)
  }
)
mod_with_break <- custom_mod(10)
expanded_break <- expand_module(mod_with_break,
                                 arg_exprs = list(x = as.symbol("x")))
# The x$size(1) call should pass through but the overall module
# still produces statements
expect_true(length(expanded_break$statements) > 0,
            info = "module with size() still produces statements")

# ===== Step 7: trace_module API =====

# Trace nn_linear
m_trace <- nn_linear(10, 5)
m_trace$eval()
x_example <- torch_randn(4, 10)

traced <- trace_module(m_trace, input = x_example)

expect_true(inherits(traced, "traced_module"),
            info = "trace_module returns traced_module")
expect_true(is.function(traced$fn),
            info = "traced$fn is a function")
expect_true(length(traced$params) > 0,
            info = "traced has parameters")
expect_equal(length(traced$graph_breaks), 0,
             info = "nn_linear traces cleanly")

# Execute and compare
result <- traced$fn(input = x_example)
ref <- m_trace(x_example)
expect_true(inherits(result, "torch_tensor"),
            info = "traced function returns tensor")
close_enough <- as.logical(torch_allclose(result, ref, atol = 1e-5))
expect_true(close_enough,
            info = "traced nn_linear matches eager execution")

# ===== Step 7b: trace_module — MLP =====

mlp_trace <- nn_sequential(
  nn_linear(10, 20),
  nn_relu(),
  nn_linear(20, 5)
)
mlp_trace$eval()
x_mlp <- torch_randn(4, 10)

traced_mlp <- trace_module(mlp_trace, input = x_mlp)
result_mlp <- traced_mlp$fn(input = x_mlp)
ref_mlp <- mlp_trace(x_mlp)

expect_true(inherits(result_mlp, "torch_tensor"),
            info = "traced MLP returns tensor")
close_mlp <- as.logical(torch_allclose(result_mlp, ref_mlp, atol = 1e-5))
expect_true(close_mlp,
            info = "traced MLP matches eager execution")

# ===== Step 7c: trace_module with layer_norm =====

ln_mod <- nn_layer_norm(20)
ln_mod$eval()
x_ln <- torch_randn(4, 20)

traced_ln <- trace_module(ln_mod, input = x_ln)
result_ln <- traced_ln$fn(input = x_ln)
ref_ln <- ln_mod(x_ln)

expect_true(inherits(result_ln, "torch_tensor"),
            info = "traced layer_norm returns tensor")
close_ln <- as.logical(torch_allclose(result_ln, ref_ln, atol = 1e-5))
expect_true(close_ln,
            info = "traced layer_norm matches eager execution")

# ===== Print method =====

out <- capture.output(print(traced))
expect_true(length(out) > 0, info = "print.traced_module produces output")
expect_true(any(grepl("Traced nn_module", out)),
            info = "print shows 'Traced nn_module' header")
expect_true(any(grepl("Parameter", out)),
            info = "print shows parameter info")
