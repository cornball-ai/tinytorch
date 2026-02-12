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

extern "C" SEXP C_transformer_decoder_layer_step(
    SEXP x_sexp,           // (batch, 1, n_state)
    SEXP weights_sexp,     // R list of 21 weight tensors
    SEXP self_cache_k_sexp,  // (batch, n_head, seq_so_far, head_dim)
    SEXP self_cache_v_sexp,
    SEXP cross_cache_k_sexp, // (batch, n_head, src_len, head_dim)
    SEXP cross_cache_v_sexp,
    SEXP n_head_sexp)        // integer
{
    try {
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
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}


// ================================================================
// Fused encoder layer: self-attention + MLP (no cross-attention, no KV cache)
// Executes one complete pre-norm encoder layer in C++ with zero R allocation.
// ================================================================

extern "C" SEXP C_transformer_encoder_layer(
    SEXP x_sexp,           // (batch, seq_len, n_state)
    SEXP weights_sexp,     // R list of 15 weight tensors
    SEXP n_head_sexp)      // integer
{
    try {
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
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}
