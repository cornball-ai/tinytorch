if (!Rtorch::is_available()) exit_file("LibTorch not available")

# ===== diffuseR (CLIP) Module Tracing =====
#
# Tests trace_module on CLIP text encoder modules from diffuseR.
# Uses inline module definitions at tiny dimensions.
# Sources ~/diffuseR/R/text_encoder.R for the actual module code.

# diffuseR uses torch R6 modules — skip until ported to Rtorch
exit_file("diffuseR not yet ported to Rtorch")

# Tiny dimensions for fast testing
embed_dim <- 64L
n_heads <- 4L
mlp_dim <- 256L
batch <- 1L
seq_len <- 10L

x <- torch_randn(batch, seq_len, embed_dim)

# ===== CLIPMLP (tanh gelu) — 0 breaks =====

clip_mlp_tanh <- CLIPMLP(embed_dim, mlp_dim, gelu_type = "tanh")
clip_mlp_tanh$eval()

traced_tanh <- trace_module(clip_mlp_tanh, x = x)

expect_equal(length(traced_tanh$graph_breaks), 0,
             info = "CLIPMLP (tanh) traces with 0 graph breaks")

expect_true(!is.null(traced_tanh$ir),
            info = "CLIPMLP (tanh) produces IR")

# Parameter capture
tanh_params <- names(traced_tanh$params)
expect_true("fc1.weight" %in% tanh_params,
            info = "CLIPMLP captures fc1.weight")
expect_true("fc1.bias" %in% tanh_params,
            info = "CLIPMLP captures fc1.bias")
expect_true("fc2.weight" %in% tanh_params,
            info = "CLIPMLP captures fc2.weight")
expect_true("fc2.bias" %in% tanh_params,
            info = "CLIPMLP captures fc2.bias")

# Correctness
ref_tanh <- clip_mlp_tanh(x)
result_tanh <- traced_tanh$fn(x = x)
expect_true(inherits(result_tanh, "torch_tensor"),
            info = "Traced CLIPMLP (tanh) returns tensor")
expect_equal(as.integer(result_tanh$shape), as.integer(ref_tanh$shape),
             info = "Traced CLIPMLP (tanh) output shape matches eager")
close_tanh <- as.logical(torch_allclose(result_tanh, ref_tanh, atol = 1e-4))
expect_true(close_tanh,
            info = "Traced CLIPMLP (tanh) matches eager (atol=1e-4)")

# ===== CLIPMLP (quick gelu) — 0 breaks =====

clip_mlp_quick <- CLIPMLP(embed_dim, mlp_dim, gelu_type = "quick")
clip_mlp_quick$eval()

traced_quick <- trace_module(clip_mlp_quick, x = x)

expect_equal(length(traced_quick$graph_breaks), 0,
             info = "CLIPMLP (quick) traces with 0 graph breaks")

# Correctness
ref_quick <- clip_mlp_quick(x)
result_quick <- traced_quick$fn(x = x)
close_quick <- as.logical(torch_allclose(result_quick, ref_quick, atol = 1e-4))
expect_true(close_quick,
            info = "Traced CLIPMLP (quick) matches eager (atol=1e-4)")

# ===== CLIPAttention — 0 breaks =====

clip_attn <- CLIPAttention(embed_dim, n_heads)
clip_attn$eval()

traced_attn <- trace_module(clip_attn, x = x)

expect_equal(length(traced_attn$graph_breaks), 0,
             info = "CLIPAttention traces with 0 graph breaks")

# Parameter capture: q_proj, k_proj, v_proj, out_proj (weight + bias each)
attn_params <- names(traced_attn$params)
expect_true("q_proj.weight" %in% attn_params,
            info = "CLIPAttention captures q_proj.weight")
expect_true("k_proj.weight" %in% attn_params,
            info = "CLIPAttention captures k_proj.weight")
expect_true("v_proj.weight" %in% attn_params,
            info = "CLIPAttention captures v_proj.weight")
expect_true("out_proj.weight" %in% attn_params,
            info = "CLIPAttention captures out_proj.weight")
expect_true("q_proj.bias" %in% attn_params,
            info = "CLIPAttention captures q_proj.bias")
expect_true("out_proj.bias" %in% attn_params,
            info = "CLIPAttention captures out_proj.bias")

# Correctness
ref_attn <- clip_attn(x)
result_attn <- traced_attn$fn(x = x)
expect_true(inherits(result_attn, "torch_tensor"),
            info = "Traced CLIPAttention returns tensor")
close_attn <- as.logical(torch_allclose(result_attn, ref_attn, atol = 1e-4))
expect_true(close_attn,
            info = "Traced CLIPAttention matches eager (atol=1e-4)")

# ===== CLIPTransformerBlock — 0 breaks =====

clip_block <- CLIPTransformerBlock(embed_dim, n_heads, mlp_dim, gelu_type = "tanh")
clip_block$eval()

traced_block <- trace_module(clip_block, x = x)

expect_equal(length(traced_block$graph_breaks), 0,
             info = "CLIPTransformerBlock traces with 0 graph breaks")

# Parameter capture: should have attention + 2 layernorms + MLP
block_params <- names(traced_block$params)
expect_true("attention.q_proj.weight" %in% block_params,
            info = "Block captures attention.q_proj.weight")
expect_true("layernorm_1.weight" %in% block_params,
            info = "Block captures layernorm_1.weight")
expect_true("layernorm_2.weight" %in% block_params,
            info = "Block captures layernorm_2.weight")
expect_true("mlp.fc1.weight" %in% block_params,
            info = "Block captures mlp.fc1.weight")
expect_true("mlp.fc2.weight" %in% block_params,
            info = "Block captures mlp.fc2.weight")

# Correctness
ref_block <- clip_block(x)
result_block <- traced_block$fn(x = x)
expect_true(inherits(result_block, "torch_tensor"),
            info = "Traced CLIPTransformerBlock returns tensor")
close_block <- as.logical(torch_allclose(result_block, ref_block, atol = 1e-4))
expect_true(close_block,
            info = "Traced CLIPTransformerBlock matches eager (atol=1e-4)")

# ===== Stacked blocks (mini text encoder, 2 layers) =====

# Build a mini encoder: 2 CLIPTransformerBlocks stacked
mini_encoder <- nn_module(
  initialize = function(embed_dim, n_heads, mlp_dim, n_layers) {
    self$n_layers <- n_layers
    self$blocks <- nn_module_list()
    for (i in seq_len(n_layers)) {
      self$blocks$append(
        CLIPTransformerBlock(embed_dim, n_heads, mlp_dim, gelu_type = "tanh")
      )
    }
    self$final_ln <- nn_layer_norm(embed_dim)
  },
  forward = function(x) {
    for (i in seq_len(self$n_layers)) {
      x <- self$blocks[[i]](x)
    }
    self$final_ln(x)
  }
)

enc <- mini_encoder(embed_dim, n_heads, mlp_dim, 2L)
enc$eval()

traced_enc <- trace_module(enc, x = x)

# Mini encoder has a for loop over nn_module_list — expect graph break
# The important thing is correctness
ref_enc <- enc(x)
result_enc <- traced_enc$fn(x = x)
expect_true(inherits(result_enc, "torch_tensor"),
            info = "Traced mini encoder returns tensor")
close_enc <- as.logical(torch_allclose(result_enc, ref_enc, atol = 1e-4))
expect_true(close_enc,
            info = "Traced stacked blocks matches eager (atol=1e-4)")

# ===== Parameter count checks =====

# CLIPMLP: fc1 (weight+bias) + fc2 (weight+bias) = 4
expect_equal(length(traced_tanh$params), 4,
             info = "CLIPMLP has 4 parameters")

# CLIPAttention: 4 projections * 2 (weight+bias) = 8
expect_equal(length(traced_attn$params), 8,
             info = "CLIPAttention has 8 parameters")

# CLIPTransformerBlock: attention(8) + layernorm_1(2) + mlp(4) + layernorm_2(2) = 16
expect_equal(length(traced_block$params), 16,
             info = "CLIPTransformerBlock has 16 parameters")

# ===== Sub-module extraction from block =====

priv <- clip_block$.__enclos_env__$private

# MLP sub-module
mlp_inst <- priv$modules_$mlp
traced_mlp_sub <- trace_module(mlp_inst, x = torch_randn(batch, seq_len, embed_dim))
expect_equal(length(traced_mlp_sub$graph_breaks), 0,
             info = "MLP sub-module traces cleanly")

ref_mlp_sub <- mlp_inst(x)
result_mlp_sub <- traced_mlp_sub$fn(x = x)
close_mlp_sub <- as.logical(torch_allclose(result_mlp_sub, ref_mlp_sub, atol = 1e-4))
expect_true(close_mlp_sub,
            info = "Standalone CLIPMLP sub-module matches eager")

# Attention sub-module
attn_inst <- priv$modules_$attention
traced_attn_sub <- trace_module(attn_inst, x = torch_randn(batch, seq_len, embed_dim))
expect_equal(length(traced_attn_sub$graph_breaks), 0,
             info = "Attention sub-module traces cleanly")

ref_attn_sub <- attn_inst(x)
result_attn_sub <- traced_attn_sub$fn(x = x)
close_attn_sub <- as.logical(torch_allclose(result_attn_sub, ref_attn_sub, atol = 1e-4))
expect_true(close_attn_sub,
            info = "Standalone CLIPAttention sub-module matches eager")
