// Fused transformer layers for Rtorch
// Execute complete pre-norm transformer layers in C++ with zero R allocation.

#include "Rtorch.h"

// Helper: extract tensor from R list by index (0-based)
static at::Tensor& list_tensor(SEXP list, int idx) {
    return *get_tensor_ptr(VECTOR_ELT(list, idx));
}

// Helper: reshape (batch, 1, n_state) -> (batch, n_head, 1, head_dim)
static at::Tensor reshape_qkv(const at::Tensor& x, int64_t batch, int64_t n_head, int64_t head_dim) {
    return x.view({batch, 1, n_head, head_dim}).transpose(1, 2);
}

// [[Rcpp::export]]
SEXP C_transformer_decoder_layer_step(
    SEXP x_sexp,           // (batch, 1, n_state)
    SEXP weights_sexp,     // R list of 21 weight tensors
    SEXP self_cache_k_sexp,  // (batch, n_head, seq_so_far, head_dim)
    SEXP self_cache_v_sexp,
    SEXP cross_cache_k_sexp, // (batch, n_head, src_len, head_dim)
    SEXP cross_cache_v_sexp,
    SEXP n_head_sexp)        // integer
{
        at::NoGradGuard no_grad;
        auto& x_in = *get_tensor_ptr(x_sexp);
        int64_t n_head = (int64_t)Rf_asInteger(n_head_sexp);
        int64_t batch = x_in.size(0);
        int64_t n_state = x_in.size(2);
        int64_t head_dim = n_state / n_head;

        // Weight indices (must match R-side packing order)
        //  0: attn_ln.weight    1: attn_ln.bias
        //  2: self_q.weight     3: self_q.bias
        //  4: self_k.weight     (no bias)
        //  5: self_v.weight     6: self_v.bias
        //  7: self_out.weight   8: self_out.bias
        //  9: cross_attn_ln.weight  10: cross_attn_ln.bias
        // 11: cross_q.weight   12: cross_q.bias
        // 13: cross_out.weight 14: cross_out.bias
        // 15: mlp_ln.weight    16: mlp_ln.bias
        // 17: fc1.weight       18: fc1.bias
        // 19: fc2.weight       20: fc2.bias

        at::Tensor x = x_in;

        // ================================================================
        // 1. Self-attention (pre-norm)
        // ================================================================
        {
            // Layer norm
            auto normed = at::layer_norm(x, {n_state},
                list_tensor(weights_sexp, 0), list_tensor(weights_sexp, 1));

            // Q, K, V projections
            auto q = at::linear(normed, list_tensor(weights_sexp, 2),
                                list_tensor(weights_sexp, 3));
            auto k = at::linear(normed, list_tensor(weights_sexp, 4));
            auto v = at::linear(normed, list_tensor(weights_sexp, 5),
                                list_tensor(weights_sexp, 6));

            // Reshape to (batch, n_head, 1, head_dim)
            q = reshape_qkv(q, batch, n_head, head_dim);
            k = reshape_qkv(k, batch, n_head, head_dim);
            v = reshape_qkv(v, batch, n_head, head_dim);

            // Concatenate with cache
            auto& cache_k = *get_tensor_ptr(self_cache_k_sexp);
            auto& cache_v = *get_tensor_ptr(self_cache_v_sexp);
            k = at::cat({cache_k, k}, /*dim=*/2);
            v = at::cat({cache_v, v}, /*dim=*/2);

            // SDPA (FlashAttention on GPU)
            auto attn_out = at::scaled_dot_product_attention(q, k, v);

            // Reshape back to (batch, 1, n_state)
            attn_out = attn_out.transpose(1, 2).contiguous().view({batch, 1, n_state});

            // Output projection
            attn_out = at::linear(attn_out, list_tensor(weights_sexp, 7),
                                  list_tensor(weights_sexp, 8));

            // Residual
            x = x + attn_out;

            // Store updated self-attention cache for return
            // (k, v are needed below)
            self_cache_k_sexp = make_tensor_sexp(new at::Tensor(k));
            self_cache_v_sexp = make_tensor_sexp(new at::Tensor(v));
        }

        // ================================================================
        // 2. Cross-attention (pre-norm)
        // ================================================================
        {
            // Layer norm
            auto normed = at::layer_norm(x, {n_state},
                list_tensor(weights_sexp, 9), list_tensor(weights_sexp, 10));

            // Q from decoder, K/V from encoder cache (read-only)
            auto q = at::linear(normed, list_tensor(weights_sexp, 11),
                                list_tensor(weights_sexp, 12));
            q = reshape_qkv(q, batch, n_head, head_dim);

            auto& cross_k = *get_tensor_ptr(cross_cache_k_sexp);
            auto& cross_v = *get_tensor_ptr(cross_cache_v_sexp);

            // SDPA
            auto attn_out = at::scaled_dot_product_attention(q, cross_k, cross_v);

            // Reshape back
            attn_out = attn_out.transpose(1, 2).contiguous().view({batch, 1, n_state});

            // Output projection
            attn_out = at::linear(attn_out, list_tensor(weights_sexp, 13),
                                  list_tensor(weights_sexp, 14));

            // Residual
            x = x + attn_out;
        }

        // ================================================================
        // 3. MLP (pre-norm)
        // ================================================================
        {
            auto normed = at::layer_norm(x, {n_state},
                list_tensor(weights_sexp, 15), list_tensor(weights_sexp, 16));

            // fc1 + gelu + fc2
            auto h = at::linear(normed, list_tensor(weights_sexp, 17),
                                list_tensor(weights_sexp, 18));
            h = at::gelu(h);
            h = at::linear(h, list_tensor(weights_sexp, 19),
                           list_tensor(weights_sexp, 20));

            // Residual
            x = x + h;
        }

        // ================================================================
        // Return: list(output, new_self_k, new_self_v)
        // ================================================================
        SEXP result = PROTECT(Rf_allocVector(VECSXP, 3));
        SET_VECTOR_ELT(result, 0, make_tensor_sexp(new at::Tensor(x)));
        SET_VECTOR_ELT(result, 1, self_cache_k_sexp);
        SET_VECTOR_ELT(result, 2, self_cache_v_sexp);

        SEXP names = PROTECT(Rf_allocVector(STRSXP, 3));
        SET_STRING_ELT(names, 0, Rf_mkChar("output"));
        SET_STRING_ELT(names, 1, Rf_mkChar("self_k"));
        SET_STRING_ELT(names, 2, Rf_mkChar("self_v"));
        Rf_setAttrib(result, R_NamesSymbol, names);

        UNPROTECT(2);
        return result;
}


// ================================================================
// Fused encoder layer: self-attention + MLP (no cross-attention, no KV cache)
// Executes one complete pre-norm encoder layer in C++ with zero R allocation.
// ================================================================

// [[Rcpp::export]]
SEXP C_transformer_encoder_layer(
    SEXP x_sexp,           // (batch, seq_len, n_state)
    SEXP weights_sexp,     // R list of 15 weight tensors
    SEXP n_head_sexp)      // integer
{
        at::NoGradGuard no_grad;
        auto& x_in = *get_tensor_ptr(x_sexp);
        int64_t n_head = (int64_t)Rf_asInteger(n_head_sexp);
        int64_t batch = x_in.size(0);
        int64_t seq_len = x_in.size(1);
        int64_t n_state = x_in.size(2);
        int64_t head_dim = n_state / n_head;

        // Weight indices (must match R-side packing order)
        //  0: attn_ln.weight    1: attn_ln.bias
        //  2: self_q.weight     3: self_q.bias
        //  4: self_k.weight     (no bias)
        //  5: self_v.weight     6: self_v.bias
        //  7: self_out.weight   8: self_out.bias
        //  9: mlp_ln.weight    10: mlp_ln.bias
        // 11: fc1.weight       12: fc1.bias
        // 13: fc2.weight       14: fc2.bias

        at::Tensor x = x_in;

        // ================================================================
        // 1. Self-attention (pre-norm)
        // ================================================================
        {
            auto normed = at::layer_norm(x, {n_state},
                list_tensor(weights_sexp, 0), list_tensor(weights_sexp, 1));

            // Q, K, V projections
            auto q = at::linear(normed, list_tensor(weights_sexp, 2),
                                list_tensor(weights_sexp, 3));
            auto k = at::linear(normed, list_tensor(weights_sexp, 4));
            auto v = at::linear(normed, list_tensor(weights_sexp, 5),
                                list_tensor(weights_sexp, 6));

            // Reshape: (batch, seq, n_state) -> (batch, n_head, seq, head_dim)
            q = q.view({batch, seq_len, n_head, head_dim}).transpose(1, 2);
            k = k.view({batch, seq_len, n_head, head_dim}).transpose(1, 2);
            v = v.view({batch, seq_len, n_head, head_dim}).transpose(1, 2);

            // SDPA (FlashAttention on GPU)
            auto attn_out = at::scaled_dot_product_attention(q, k, v);

            // Reshape back: (batch, n_head, seq, head_dim) -> (batch, seq, n_state)
            attn_out = attn_out.transpose(1, 2).contiguous().view({batch, seq_len, n_state});

            // Output projection + residual
            attn_out = at::linear(attn_out, list_tensor(weights_sexp, 7),
                                  list_tensor(weights_sexp, 8));
            x = x + attn_out;
        }

        // ================================================================
        // 2. MLP (pre-norm)
        // ================================================================
        {
            auto normed = at::layer_norm(x, {n_state},
                list_tensor(weights_sexp, 9), list_tensor(weights_sexp, 10));

            auto h = at::linear(normed, list_tensor(weights_sexp, 11),
                                list_tensor(weights_sexp, 12));
            h = at::gelu(h);
            h = at::linear(h, list_tensor(weights_sexp, 13),
                           list_tensor(weights_sexp, 14));

            x = x + h;
        }

        return make_tensor_sexp(new at::Tensor(x));
}


// ================================================================
// Prepare cross-attention caches: project encoder output through
// each decoder layer's cross-attention K/V weights.
// ================================================================

// [[Rcpp::export]]
SEXP C_prepare_cross_caches(
    SEXP encoder_output_sexp,  // (batch, src_len, n_state)
    SEXP cross_kv_weights,     // list of N lists, each [k_weight, v_weight, v_bias]
    SEXP n_head_sexp)
{
    at::NoGradGuard no_grad;
    auto& enc = *get_tensor_ptr(encoder_output_sexp);
    int64_t n_head = (int64_t)Rf_asInteger(n_head_sexp);
    int64_t n_layers = Rf_length(cross_kv_weights);
    int64_t batch = enc.size(0);
    int64_t src_len = enc.size(1);
    int64_t n_state = enc.size(2);
    int64_t head_dim = n_state / n_head;

    SEXP k_list = PROTECT(Rf_allocVector(VECSXP, n_layers));
    SEXP v_list = PROTECT(Rf_allocVector(VECSXP, n_layers));

    for (int64_t i = 0; i < n_layers; i++) {
        SEXP lw = VECTOR_ELT(cross_kv_weights, i);
        auto& k_weight = list_tensor(lw, 0);
        auto& v_weight = list_tensor(lw, 1);
        auto& v_bias   = list_tensor(lw, 2);

        auto k = at::linear(enc, k_weight);
        auto v = at::linear(enc, v_weight, v_bias);

        k = k.view({batch, src_len, n_head, head_dim}).transpose(1, 2);
        v = v.view({batch, src_len, n_head, head_dim}).transpose(1, 2);

        SET_VECTOR_ELT(k_list, i, make_tensor_sexp(new at::Tensor(k)));
        SET_VECTOR_ELT(v_list, i, make_tensor_sexp(new at::Tensor(v)));
    }

    SEXP result = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(result, 0, k_list);
    SET_VECTOR_ELT(result, 1, v_list);

    SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, Rf_mkChar("k"));
    SET_STRING_ELT(names, 1, Rf_mkChar("v"));
    Rf_setAttrib(result, R_NamesSymbol, names);

    UNPROTECT(4);
    return result;
}


// ================================================================
// Fused encoder forward: conv stem + all layers + final LN
// Executes the entire encoder in a single .Call().
// ================================================================

// [[Rcpp::export]]
SEXP C_encoder_forward(
    SEXP mel_sexp,           // (batch, n_mels, n_frames)
    SEXP global_weights,     // list: [conv1_w, conv1_b, conv2_w, conv2_b, pos_emb, ln_w, ln_b]
    SEXP layer_weights,      // list of lists: N layers x 15 weights each
    SEXP n_head_sexp,        // integer
    SEXP n_ctx_sexp)         // integer: max context length for truncation
{
    at::NoGradGuard no_grad;
    auto& mel = *get_tensor_ptr(mel_sexp);
    int64_t n_head = (int64_t)Rf_asInteger(n_head_sexp);
    int64_t n_ctx = (int64_t)Rf_asInteger(n_ctx_sexp);
    int64_t n_layers = Rf_length(layer_weights);

    // Global weights
    auto& conv1_w = list_tensor(global_weights, 0);
    auto& conv1_b = list_tensor(global_weights, 1);
    auto& conv2_w = list_tensor(global_weights, 2);
    auto& conv2_b = list_tensor(global_weights, 3);
    auto& pos_emb = list_tensor(global_weights, 4);
    auto& ln_w    = list_tensor(global_weights, 5);
    auto& ln_b    = list_tensor(global_weights, 6);

    // 1. Conv stem: conv1 + gelu, conv2 + gelu
    auto x = at::gelu(at::conv1d(mel, conv1_w, conv1_b, /*stride=*/1, /*padding=*/1));
    x = at::gelu(at::conv1d(x, conv2_w, conv2_b, /*stride=*/2, /*padding=*/1));

    // 2. Permute: (batch, n_state, n_frames/2) -> (batch, n_frames/2, n_state)
    x = x.permute({0, 2, 1});
    int64_t batch = x.size(0);
    int64_t seq_len = x.size(1);
    int64_t n_state = x.size(2);
    int64_t head_dim = n_state / n_head;

    // 3. Truncate if longer than max context
    if (seq_len > n_ctx) {
        x = x.slice(1, 0, n_ctx);
        seq_len = n_ctx;
    }

    // 4. Add positional embedding (slice to seq_len)
    x = x + pos_emb.slice(0, 0, seq_len).unsqueeze(0);

    // 5. Transformer layers
    for (int64_t i = 0; i < n_layers; i++) {
        SEXP lw = VECTOR_ELT(layer_weights, i);

        // Self-attention (pre-norm)
        {
            auto normed = at::layer_norm(x, {n_state},
                list_tensor(lw, 0), list_tensor(lw, 1));

            auto q = at::linear(normed, list_tensor(lw, 2), list_tensor(lw, 3));
            auto k = at::linear(normed, list_tensor(lw, 4));
            auto v = at::linear(normed, list_tensor(lw, 5), list_tensor(lw, 6));

            q = q.view({batch, seq_len, n_head, head_dim}).transpose(1, 2);
            k = k.view({batch, seq_len, n_head, head_dim}).transpose(1, 2);
            v = v.view({batch, seq_len, n_head, head_dim}).transpose(1, 2);

            auto attn_out = at::scaled_dot_product_attention(q, k, v);
            attn_out = attn_out.transpose(1, 2).contiguous()
                              .view({batch, seq_len, n_state});
            attn_out = at::linear(attn_out, list_tensor(lw, 7), list_tensor(lw, 8));
            x = x + attn_out;
        }

        // MLP (pre-norm)
        {
            auto normed = at::layer_norm(x, {n_state},
                list_tensor(lw, 9), list_tensor(lw, 10));
            auto h = at::linear(normed, list_tensor(lw, 11), list_tensor(lw, 12));
            h = at::gelu(h);
            h = at::linear(h, list_tensor(lw, 13), list_tensor(lw, 14));
            x = x + h;
        }
    }

    // 6. Final layer norm
    x = at::layer_norm(x, {n_state}, ln_w, ln_b);

    return make_tensor_sexp(new at::Tensor(x));
}


// ================================================================
// Fused decoder forward step: embedding + all layers + LN + logits + argmax
// Executes the entire decoder forward pass in a single .Call() per token.
// ================================================================

// [[Rcpp::export]]
SEXP C_decoder_forward_step(
    SEXP token_ids_sexp,      // (batch, seq_len) int64 — 0-indexed token IDs
    SEXP global_weights,      // list: [token_emb, pos_emb, final_ln_w, final_ln_b]
    SEXP layer_weights,       // list of lists: N layers x 21 weights each
    SEXP self_cache_k_list,   // list of N tensors (or NULL for prefill)
    SEXP self_cache_v_list,
    SEXP cross_cache_k_list,  // list of N tensors
    SEXP cross_cache_v_list,
    SEXP n_head_sexp,
    SEXP offset_sexp)         // position offset for positional embedding
{
    at::NoGradGuard no_grad;
    auto& token_ids = *get_tensor_ptr(token_ids_sexp);
    int64_t n_head = (int64_t)Rf_asInteger(n_head_sexp);
    int64_t offset = (int64_t)Rf_asInteger(offset_sexp);
    int64_t n_layers = Rf_length(layer_weights);
    int64_t batch = token_ids.size(0);
    int64_t seq_len = token_ids.size(1);
    bool is_prefill = (self_cache_k_list == R_NilValue);

    // Global weights
    auto& token_emb_w = list_tensor(global_weights, 0);
    auto& pos_emb_w   = list_tensor(global_weights, 1);
    auto& final_ln_w  = list_tensor(global_weights, 2);
    auto& final_ln_b  = list_tensor(global_weights, 3);

    // 1. Token embedding (0-indexed, at::embedding is 0-based)
    auto x = at::embedding(token_emb_w, token_ids);
    int64_t n_state = x.size(2);
    int64_t head_dim = n_state / n_head;

    // 2. Positional embedding
    auto positions = at::arange(offset, offset + seq_len,
        at::TensorOptions().dtype(at::kLong).device(x.device()));
    x = x + at::embedding(pos_emb_w, positions).unsqueeze(0);

    // Allocate output self-cache lists
    SEXP new_self_k_list = PROTECT(Rf_allocVector(VECSXP, n_layers));
    SEXP new_self_v_list = PROTECT(Rf_allocVector(VECSXP, n_layers));

    // 3. Process each decoder layer
    for (int64_t i = 0; i < n_layers; i++) {
        SEXP lw = VECTOR_ELT(layer_weights, i);

        // ---- Self-attention (pre-norm) ----
        {
            auto normed = at::layer_norm(x, {n_state},
                list_tensor(lw, 0), list_tensor(lw, 1));

            auto q = at::linear(normed, list_tensor(lw, 2), list_tensor(lw, 3));
            auto k = at::linear(normed, list_tensor(lw, 4));
            auto v = at::linear(normed, list_tensor(lw, 5), list_tensor(lw, 6));

            q = q.view({batch, seq_len, n_head, head_dim}).transpose(1, 2);
            k = k.view({batch, seq_len, n_head, head_dim}).transpose(1, 2);
            v = v.view({batch, seq_len, n_head, head_dim}).transpose(1, 2);

            if (!is_prefill) {
                auto& cache_k = *get_tensor_ptr(VECTOR_ELT(self_cache_k_list, i));
                auto& cache_v = *get_tensor_ptr(VECTOR_ELT(self_cache_v_list, i));
                k = at::cat({cache_k, k}, 2);
                v = at::cat({cache_v, v}, 2);
            }

            // SDPA — use is_causal for prefill, no mask for incremental
            at::Tensor attn_out;
            if (is_prefill && seq_len > 1) {
                attn_out = at::scaled_dot_product_attention(
                    q, k, v, /*attn_mask=*/c10::nullopt, /*dropout_p=*/0.0,
                    /*is_causal=*/true);
            } else {
                attn_out = at::scaled_dot_product_attention(q, k, v);
            }

            attn_out = attn_out.transpose(1, 2).contiguous()
                              .view({batch, seq_len, n_state});
            attn_out = at::linear(attn_out, list_tensor(lw, 7), list_tensor(lw, 8));
            x = x + attn_out;

            SET_VECTOR_ELT(new_self_k_list, i, make_tensor_sexp(new at::Tensor(k)));
            SET_VECTOR_ELT(new_self_v_list, i, make_tensor_sexp(new at::Tensor(v)));
        }

        // ---- Cross-attention (pre-norm) ----
        {
            auto normed = at::layer_norm(x, {n_state},
                list_tensor(lw, 9), list_tensor(lw, 10));

            auto q = at::linear(normed, list_tensor(lw, 11), list_tensor(lw, 12));
            q = q.view({batch, seq_len, n_head, head_dim}).transpose(1, 2);

            auto& cross_k = *get_tensor_ptr(VECTOR_ELT(cross_cache_k_list, i));
            auto& cross_v = *get_tensor_ptr(VECTOR_ELT(cross_cache_v_list, i));

            auto attn_out = at::scaled_dot_product_attention(q, cross_k, cross_v);
            attn_out = attn_out.transpose(1, 2).contiguous()
                              .view({batch, seq_len, n_state});
            attn_out = at::linear(attn_out, list_tensor(lw, 13), list_tensor(lw, 14));
            x = x + attn_out;
        }

        // ---- MLP (pre-norm) ----
        {
            auto normed = at::layer_norm(x, {n_state},
                list_tensor(lw, 15), list_tensor(lw, 16));
            auto h = at::linear(normed, list_tensor(lw, 17), list_tensor(lw, 18));
            h = at::gelu(h);
            h = at::linear(h, list_tensor(lw, 19), list_tensor(lw, 20));
            x = x + h;
        }
    }

    // 4. Final layer norm
    x = at::layer_norm(x, {n_state}, final_ln_w, final_ln_b);

    // 5. Logits projection (tie weights with token embedding)
    auto logits = at::matmul(x, token_emb_w.t());

    // 6. Argmax on last position
    auto last_logits = logits.select(1, seq_len - 1);
    int token_id = (int)last_logits.argmax(-1).item<int64_t>();

    // Return: list(token_id, self_cache_k, self_cache_v, cross_cache_k, cross_cache_v)
    SEXP result = PROTECT(Rf_allocVector(VECSXP, 5));
    SET_VECTOR_ELT(result, 0, Rf_ScalarInteger(token_id));
    SET_VECTOR_ELT(result, 1, new_self_k_list);
    SET_VECTOR_ELT(result, 2, new_self_v_list);
    SET_VECTOR_ELT(result, 3, cross_cache_k_list);
    SET_VECTOR_ELT(result, 4, cross_cache_v_list);

    SEXP names = PROTECT(Rf_allocVector(STRSXP, 5));
    SET_STRING_ELT(names, 0, Rf_mkChar("token_id"));
    SET_STRING_ELT(names, 1, Rf_mkChar("self_cache_k"));
    SET_STRING_ELT(names, 2, Rf_mkChar("self_cache_v"));
    SET_STRING_ELT(names, 3, Rf_mkChar("cross_cache_k"));
    SET_STRING_ELT(names, 4, Rf_mkChar("cross_cache_v"));
    Rf_setAttrib(result, R_NamesSymbol, names);

    UNPROTECT(4);
    return result;
}


// ================================================================
// Greedy decode: entire autoregressive loop in C++.
// One .Call() from R — no per-token boundary crossings.
// ================================================================

// [[Rcpp::export]]
SEXP C_greedy_decode(
    SEXP initial_tokens_sexp,    // integer vector of initial token IDs (0-indexed)
    SEXP global_weights,         // list: [token_emb, pos_emb, final_ln_w, final_ln_b]
    SEXP layer_weights,          // list of lists: N layers x 21 weights each
    SEXP cross_cache_k_list,     // list of N tensors
    SEXP cross_cache_v_list,     // list of N tensors
    SEXP n_head_sexp,
    SEXP max_length_sexp,
    SEXP eot_token_sexp)         // end-of-text token ID
{
    at::NoGradGuard no_grad;

    int64_t n_head = (int64_t)Rf_asInteger(n_head_sexp);
    int64_t max_length = (int64_t)Rf_asInteger(max_length_sexp);
    int64_t eot_token = (int64_t)Rf_asInteger(eot_token_sexp);
    int64_t n_layers = Rf_length(layer_weights);

    // Global weights
    auto& token_emb_w = list_tensor(global_weights, 0);
    auto& pos_emb_w   = list_tensor(global_weights, 1);
    auto& final_ln_w  = list_tensor(global_weights, 2);
    auto& final_ln_b  = list_tensor(global_weights, 3);

    int64_t n_state = token_emb_w.size(1);
    int64_t head_dim = n_state / n_head;
    auto device = token_emb_w.device();

    // Copy initial tokens to output vector
    int* init_ptr = INTEGER(initial_tokens_sexp);
    int n_init = Rf_length(initial_tokens_sexp);
    std::vector<int> generated(init_ptr, init_ptr + n_init);

    // Read cross-attention caches (read-only, no copies)
    std::vector<at::Tensor> cross_k(n_layers), cross_v(n_layers);
    for (int64_t i = 0; i < n_layers; i++) {
        cross_k[i] = *get_tensor_ptr(VECTOR_ELT(cross_cache_k_list, i));
        cross_v[i] = *get_tensor_ptr(VECTOR_ELT(cross_cache_v_list, i));
    }

    // Self-attention KV caches (grow each step)
    std::vector<at::Tensor> self_k(n_layers), self_v(n_layers);

    int64_t offset = 0;

    // One decoder forward step (prefill or incremental)
    auto forward_step = [&](const at::Tensor& tokens, bool is_prefill) -> int64_t {
        int64_t seq_len = tokens.size(1);

        // Token + positional embedding
        auto x = at::embedding(token_emb_w, tokens);
        auto positions = at::arange(offset, offset + seq_len,
            at::TensorOptions().dtype(at::kLong).device(device));
        x = x + at::embedding(pos_emb_w, positions).unsqueeze(0);

        for (int64_t i = 0; i < n_layers; i++) {
            SEXP lw = VECTOR_ELT(layer_weights, i);

            // Self-attention (pre-norm)
            {
                auto normed = at::layer_norm(x, {n_state},
                    list_tensor(lw, 0), list_tensor(lw, 1));
                auto q = at::linear(normed, list_tensor(lw, 2), list_tensor(lw, 3));
                auto k = at::linear(normed, list_tensor(lw, 4));
                auto v = at::linear(normed, list_tensor(lw, 5), list_tensor(lw, 6));

                q = q.view({1, seq_len, n_head, head_dim}).transpose(1, 2);
                k = k.view({1, seq_len, n_head, head_dim}).transpose(1, 2);
                v = v.view({1, seq_len, n_head, head_dim}).transpose(1, 2);

                if (!is_prefill) {
                    k = at::cat({self_k[i], k}, 2);
                    v = at::cat({self_v[i], v}, 2);
                }
                self_k[i] = k;
                self_v[i] = v;

                at::Tensor attn_out;
                if (is_prefill && seq_len > 1) {
                    attn_out = at::scaled_dot_product_attention(
                        q, k, v, c10::nullopt, 0.0, true);
                } else {
                    attn_out = at::scaled_dot_product_attention(q, k, v);
                }

                attn_out = attn_out.transpose(1, 2).contiguous()
                                  .view({1, seq_len, n_state});
                attn_out = at::linear(attn_out, list_tensor(lw, 7), list_tensor(lw, 8));
                x = x + attn_out;
            }

            // Cross-attention (pre-norm)
            {
                auto normed = at::layer_norm(x, {n_state},
                    list_tensor(lw, 9), list_tensor(lw, 10));
                auto q = at::linear(normed, list_tensor(lw, 11), list_tensor(lw, 12));
                q = q.view({1, seq_len, n_head, head_dim}).transpose(1, 2);

                auto attn_out = at::scaled_dot_product_attention(q, cross_k[i], cross_v[i]);
                attn_out = attn_out.transpose(1, 2).contiguous()
                                  .view({1, seq_len, n_state});
                attn_out = at::linear(attn_out, list_tensor(lw, 13), list_tensor(lw, 14));
                x = x + attn_out;
            }

            // MLP (pre-norm)
            {
                auto normed = at::layer_norm(x, {n_state},
                    list_tensor(lw, 15), list_tensor(lw, 16));
                auto h = at::linear(normed, list_tensor(lw, 17), list_tensor(lw, 18));
                h = at::gelu(h);
                h = at::linear(h, list_tensor(lw, 19), list_tensor(lw, 20));
                x = x + h;
            }
        }

        // Final layer norm + logits + argmax
        x = at::layer_norm(x, {n_state}, final_ln_w, final_ln_b);
        auto logits = at::matmul(x, token_emb_w.t());
        auto last_logits = logits.select(1, seq_len - 1);
        int64_t token_id = last_logits.argmax(-1).item<int64_t>();

        offset += seq_len;
        return token_id;
    };

    // Build initial token tensor: (1, n_init) int64
    auto token_tensor = at::tensor(
        std::vector<int64_t>(generated.begin(), generated.end()),
        at::TensorOptions().dtype(at::kLong).device(device)
    ).unsqueeze(0);

    // Prefill: process all initial tokens at once
    int64_t next_token = forward_step(token_tensor, true);

    // Incremental decode loop (entirely in C++)
    while (next_token != eot_token && (int64_t)generated.size() < max_length) {
        generated.push_back((int)next_token);

        auto single_token = at::tensor({next_token},
            at::TensorOptions().dtype(at::kLong).device(device)
        ).unsqueeze(0);

        next_token = forward_step(single_token, false);
    }

    // Return integer vector
    SEXP result = PROTECT(Rf_allocVector(INTSXP, generated.size()));
    std::memcpy(INTEGER(result), generated.data(), generated.size() * sizeof(int));
    UNPROTECT(1);
    return result;
}
