# ===== Chatterbox (Llama) Module Tracing =====
#
# Tests trace_module on Llama-style modules from chatterbox.
# Uses inline module definitions at tiny dimensions.
# Sources ~/chatterbox/R/llama.R for the actual module code.

# chatterbox uses torch R6 modules — skip until ported to Rtorch
exit_file("chatterbox not yet ported to Rtorch")

# Tiny dimensions for fast testing
hidden_size <- 64L
intermediate_size <- 256L
n_heads <- 4L
head_dim_llama <- hidden_size %/% n_heads
batch <- 1L
seq_len <- 16L

config_tiny <- list(
  vocab_size = 8L,
  max_position_embeddings = 128L,
  hidden_size = hidden_size,
  intermediate_size = intermediate_size,
  num_hidden_layers = 2L,
  num_attention_heads = n_heads,
  num_key_value_heads = n_heads,
  head_dim = head_dim_llama,
  hidden_act = "silu",
  attention_bias = FALSE,
  attention_dropout = 0.0,
  mlp_bias = FALSE,
  rms_norm_eps = 1e-5,
  rope_theta = 500000.0,
  rope_scaling = NULL
)

# ===== llama_rms_norm — should trace cleanly =====

rms <- llama_rms_norm(hidden_size)
rms$eval()
x_rms <- torch_randn(batch, seq_len, hidden_size)

traced_rms <- trace_module(rms, x = x_rms)

expect_equal(length(traced_rms$graph_breaks), 0,
             info = "llama_rms_norm traces with 0 graph breaks")

expect_true(!is.null(traced_rms$ir),
            info = "llama_rms_norm produces IR")

# Parameter capture
rms_params <- names(traced_rms$params)
expect_true("weight" %in% rms_params,
            info = "RMSNorm captures weight parameter")

# Correctness
ref_rms <- rms(x_rms)
result_rms <- traced_rms$fn(x = x_rms)
expect_true(inherits(result_rms, "torch_tensor"),
            info = "Traced RMSNorm returns tensor")
expect_equal(as.integer(result_rms$shape), as.integer(ref_rms$shape),
             info = "Traced RMSNorm output shape matches eager")
close_rms <- as.logical(torch_allclose(result_rms, ref_rms, atol = 1e-4))
expect_true(close_rms,
            info = "Traced llama_rms_norm matches eager (atol=1e-4)")

# ===== llama_mlp — 1 graph break expected (self$act_fn) =====

mlp <- llama_mlp(config_tiny)
mlp$eval()
x_mlp <- torch_randn(batch, seq_len, hidden_size)

traced_mlp <- trace_module(mlp, x = x_mlp)

expect_equal(length(traced_mlp$graph_breaks), 1,
             info = "llama_mlp traces with 1 graph break (self$act_fn)")

# Parameter capture: gate_proj, up_proj, down_proj (all bias=FALSE)
mlp_params <- names(traced_mlp$params)
expect_true("gate_proj.weight" %in% mlp_params,
            info = "MLP captures gate_proj.weight")
expect_true("up_proj.weight" %in% mlp_params,
            info = "MLP captures up_proj.weight")
expect_true("down_proj.weight" %in% mlp_params,
            info = "MLP captures down_proj.weight")
expect_equal(sum(grepl("\\.bias$", mlp_params)), 0,
             info = "MLP has no bias parameters (mlp_bias=FALSE)")

# Correctness (via fallback eval)
ref_mlp <- mlp(x_mlp)
result_mlp <- traced_mlp$fn(x = x_mlp)
expect_true(inherits(result_mlp, "torch_tensor"),
            info = "Traced MLP returns tensor")
close_mlp <- as.logical(torch_allclose(result_mlp, ref_mlp, atol = 1e-4))
expect_true(close_mlp,
            info = "Traced llama_mlp matches eager (atol=1e-4)")

# ===== llama_decoder_layer — 2 graph breaks expected =====

dec_layer <- llama_decoder_layer(config_tiny, 0L)
dec_layer$eval()
x_dec <- torch_randn(batch, seq_len, hidden_size)
pos_ids <- torch_arange(0, seq_len - 1, dtype = torch_long())$unsqueeze(1)
rope <- compute_rope_frequencies(head_dim_llama, 128L)

traced_dec <- trace_module(
  dec_layer,
  hidden_states = x_dec,
  position_ids = pos_ids,
  rope_cos = rope$cos,
  rope_sin = rope$sin
)

expect_true(length(traced_dec$graph_breaks) >= 2,
            info = "llama_decoder_layer has at least 2 graph breaks")

# Parameter capture: should have norm weights + attention + mlp projections
dec_params <- names(traced_dec$params)
expect_true("input_layernorm.weight" %in% dec_params,
            info = "Decoder layer captures input_layernorm.weight")
expect_true("post_attention_layernorm.weight" %in% dec_params,
            info = "Decoder layer captures post_attention_layernorm.weight")
expect_true("self_attn.q_proj.weight" %in% dec_params,
            info = "Decoder layer captures self_attn.q_proj.weight")
expect_true("self_attn.o_proj.weight" %in% dec_params,
            info = "Decoder layer captures self_attn.o_proj.weight")
expect_true("mlp.gate_proj.weight" %in% dec_params,
            info = "Decoder layer captures mlp.gate_proj.weight")

# Correctness (via fallback eval — returns list)
ref_dec <- dec_layer(
  hidden_states = x_dec,
  position_ids = pos_ids,
  rope_cos = rope$cos,
  rope_sin = rope$sin
)
result_dec <- traced_dec$fn(
  hidden_states = x_dec,
  position_ids = pos_ids,
  rope_cos = rope$cos,
  rope_sin = rope$sin
)

# Extract hidden_states from list return
ref_hs <- if (is.list(ref_dec)) ref_dec$hidden_states else ref_dec
result_hs <- if (is.list(result_dec)) result_dec$hidden_states else result_dec

expect_true(inherits(result_hs, "torch_tensor"),
            info = "Traced decoder layer returns tensor (hidden_states)")
close_dec <- as.logical(torch_allclose(result_hs, ref_hs, atol = 1e-4))
expect_true(close_dec,
            info = "Traced llama_decoder_layer matches eager (atol=1e-4)")

# ===== Sub-module extraction =====

# Access sub-modules from decoder layer
priv <- dec_layer$.__enclos_env__$private

# Input layernorm
ln_inst <- priv$modules_$input_layernorm
traced_ln <- trace_module(ln_inst, x = torch_randn(batch, seq_len, hidden_size))
expect_equal(length(traced_ln$graph_breaks), 0,
             info = "Input layernorm traces cleanly as standalone")

ref_ln <- ln_inst(x_dec)
result_ln <- traced_ln$fn(x = x_dec)
close_ln <- as.logical(torch_allclose(result_ln, ref_ln, atol = 1e-5))
expect_true(close_ln,
            info = "Standalone input_layernorm matches eager")

# MLP sub-module
mlp_inst <- priv$modules_$mlp
traced_mlp_sub <- trace_module(mlp_inst, x = torch_randn(batch, seq_len, hidden_size))
expect_equal(length(traced_mlp_sub$graph_breaks), 1,
             info = "MLP sub-module has 1 graph break (self$act_fn)")

ref_mlp_sub <- mlp_inst(x_dec)
result_mlp_sub <- traced_mlp_sub$fn(x = x_dec)
close_mlp_sub <- as.logical(torch_allclose(result_mlp_sub, ref_mlp_sub, atol = 1e-4))
expect_true(close_mlp_sub,
            info = "Standalone MLP sub-module matches eager")

# ===== Parameter count checks =====

# RMSNorm: 1 weight
expect_equal(length(traced_rms$params), 1,
             info = "RMSNorm has 1 parameter (weight)")

# MLP: 3 weights (gate_proj, up_proj, down_proj — no biases)
expect_equal(length(traced_mlp$params), 3,
             info = "MLP has 3 parameters (3 projection weights, no biases)")

# Decoder layer: 2 norms (1 each) + 4 attn projections (1 each) + 3 mlp projections = 9
expect_true(length(traced_dec$params) >= 9,
            info = "Decoder layer has at least 9 parameters")
