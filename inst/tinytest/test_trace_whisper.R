if (!tinytorch::is_available()) exit_file("LibTorch not available")

# ===== Whisper Encoder Layer Tracing =====
#
# Tests trace_module on realistic Whisper-style encoder layers.
# Verifies that feedforward paths trace cleanly while attention
# paths produce expected graph breaks.

# Whisper-tiny dimensions
n_state  <- 384L
n_head   <- 6L
head_dim <- n_state %/% n_head  # 64
n_ctx    <- 100L  # shorter than real (1500) for test speed

# ===== FFN-only encoder layer (fully traceable) =====

whisper_ffn_layer <- nn_module(
  initialize = function(n_state) {
    self$mlp_ln <- nn_layer_norm(n_state)
    self$mlp <- nn_sequential(
      nn_linear(n_state, 4L * n_state),
      nn_gelu(),
      nn_linear(4L * n_state, n_state)
    )
  },
  forward = function(x) {
    x + self$mlp(self$mlp_ln(x))
  }
)

ffn_layer <- whisper_ffn_layer(n_state)
ffn_layer$eval()
x <- torch_randn(n_ctx, n_state)

# Trace
traced_ffn <- trace_module(ffn_layer, x = x)

expect_equal(length(traced_ffn$graph_breaks), 0,
             info = "Whisper FFN layer traces with 0 graph breaks")

# Should capture layer_norm params + 2 linear layers
param_names <- names(traced_ffn$params)
expect_true("mlp_ln.weight" %in% param_names,
            info = "FFN captures mlp_ln.weight")
expect_true("mlp_ln.bias" %in% param_names,
            info = "FFN captures mlp_ln.bias")
expect_true("mlp.0.weight" %in% param_names,
            info = "FFN captures mlp.0.weight (first linear)")
expect_true("mlp.0.bias" %in% param_names,
            info = "FFN captures mlp.0.bias (first linear)")
expect_true("mlp.2.weight" %in% param_names,
            info = "FFN captures mlp.2.weight (second linear)")
expect_true("mlp.2.bias" %in% param_names,
            info = "FFN captures mlp.2.bias (second linear)")

# Should produce IR
expect_true(!is.null(traced_ffn$ir),
            info = "FFN layer produces IR graph")

# Correctness: traced result matches eager
result <- traced_ffn$fn(x = x)
ref <- ffn_layer(x)

expect_true(inherits(result, "torch_tensor"),
            info = "Traced FFN returns tensor")
expect_equal(as.integer(result$shape), as.integer(ref$shape),
             info = "Traced FFN output shape matches eager")
close <- as.logical(torch_allclose(result, ref, atol = 1e-4))
expect_true(close,
            info = "Traced FFN output matches eager (atol=1e-4)")

# ===== Full encoder layer with attention (graph breaks expected) =====

whisper_encoder_layer <- nn_module(
  initialize = function(n_state, n_head) {
    self$n_head <- n_head
    self$n_state <- n_state
    self$head_dim <- n_state %/% n_head

    # Attention
    self$attn_ln <- nn_layer_norm(n_state)
    self$query <- nn_linear(n_state, n_state)
    self$key <- nn_linear(n_state, n_state, bias = FALSE)
    self$value <- nn_linear(n_state, n_state)
    self$out <- nn_linear(n_state, n_state)

    # FFN
    self$mlp_ln <- nn_layer_norm(n_state)
    self$mlp <- nn_sequential(
      nn_linear(n_state, 4L * n_state),
      nn_gelu(),
      nn_linear(4L * n_state, n_state)
    )
  },
  forward = function(x) {
    # Attention path: query/key/value projections then reshape
    normed <- self$attn_ln(x)
    q <- self$query(normed)
    k <- self$key(normed)
    v <- self$value(normed)

    # x$size() returns runtime shape — passes through for eval, prevents IR
    n <- x$size(1)
    q <- q$view(c(n, self$n_head, self$head_dim))$transpose(1L, 2L)
    k <- k$view(c(n, self$n_head, self$head_dim))$transpose(1L, 2L)
    v <- v$view(c(n, self$n_head, self$head_dim))$transpose(1L, 2L)

    # Attention scores
    scale <- self$head_dim ^ (-0.5)
    scores <- q$matmul(k$transpose(2L, 3L)) * scale
    attn_weights <- nnf_softmax(scores, dim = -1L)
    attn_out <- attn_weights$matmul(v)

    # Reshape back
    attn_out <- attn_out$transpose(1L, 2L)$contiguous()$view(c(n, self$n_state))
    attn_out <- self$out(attn_out)

    # Residual + FFN
    x <- x + attn_out
    x + self$mlp(self$mlp_ln(x))
  }
)

full_layer <- whisper_encoder_layer(n_state, n_head)
full_layer$eval()
x_full <- torch_randn(n_ctx, n_state)

traced_full <- trace_module(full_layer, x = x_full)

# With NULL bias fix, self$bias resolves as scalar NULL instead of unknown.
# x$size() passes through as a tensor method (no graph break at AST level).
# The full layer now traces cleanly — IR lowering may fail on x$size()
# but fallback eval handles it.
expect_equal(length(traced_full$graph_breaks), 0,
             info = "Full encoder layer traces with 0 graph breaks")

# Should capture all parameters
full_params <- names(traced_full$params)
expect_true("attn_ln.weight" %in% full_params,
            info = "Full layer captures attn_ln params")
expect_true("query.weight" %in% full_params,
            info = "Full layer captures query projection params")
expect_true("mlp.0.weight" %in% full_params,
            info = "Full layer captures FFN params")

# Correctness: traced result matches eager (via fallback eval)
ref_full <- full_layer(x_full)
result_full <- traced_full$fn(x = x_full)
expect_true(inherits(result_full, "torch_tensor"),
            info = "Traced full layer returns tensor")
close_full <- as.logical(torch_allclose(result_full, ref_full, atol = 1e-4))
expect_true(close_full,
            info = "Traced full encoder layer matches eager (atol=1e-4)")

# ===== Standalone sub-module tracing =====

# Even with graph breaks at the full layer level,
# individual sub-modules should trace cleanly.

# Layer norm — access sub-module via private$modules_
priv <- full_layer$.__enclos_env__$private
ln_inst <- priv$modules_$mlp_ln
traced_ln <- trace_module(ln_inst, input = torch_randn(n_ctx, n_state))

expect_equal(length(traced_ln$graph_breaks), 0,
             info = "Layer norm traces cleanly as standalone")

ref_ln <- ln_inst(x_full)
result_ln <- traced_ln$fn(input = x_full)
close_ln <- as.logical(torch_allclose(result_ln, ref_ln, atol = 1e-5))
expect_true(close_ln,
            info = "Standalone layer norm matches eager")

# Query projection (nn_linear)
query_inst <- priv$modules_$query
traced_q <- trace_module(query_inst, input = torch_randn(n_ctx, n_state))

expect_equal(length(traced_q$graph_breaks), 0,
             info = "Query projection traces cleanly")

ref_q <- query_inst(x_full)
result_q <- traced_q$fn(input = x_full)
close_q <- as.logical(torch_allclose(result_q, ref_q, atol = 1e-5))
expect_true(close_q,
            info = "Standalone query projection matches eager")

# FFN sequential
mlp_inst <- priv$modules_$mlp
traced_mlp <- trace_module(mlp_inst, input = torch_randn(n_ctx, n_state))

expect_equal(length(traced_mlp$graph_breaks), 0,
             info = "FFN sequential traces cleanly")

ref_mlp <- mlp_inst(x_full)
result_mlp <- traced_mlp$fn(input = x_full)
close_mlp <- as.logical(torch_allclose(result_mlp, ref_mlp, atol = 1e-4))
expect_true(close_mlp,
            info = "Standalone FFN sequential matches eager")

# ===== Print method =====

out <- capture.output(print(traced_ffn))
expect_true(any(grepl("mlp_ln\\.weight", out)),
            info = "Print shows dotted parameter names")
expect_true(any(grepl("torch_layer_norm", out)),
            info = "Print shows expanded torch_layer_norm call")
expect_true(any(grepl("torch_linear", out)),
            info = "Print shows expanded torch_linear calls")
