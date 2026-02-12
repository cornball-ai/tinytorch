#include "Rtorch.h"

// ---- Activation functions ----

// [[Rcpp::export]]
at::Tensor C_nnf_silu(at::Tensor self) { return at::silu(self); }

// [[Rcpp::export]]
at::Tensor C_nnf_gelu(at::Tensor self, SEXP approximate_sexp) {
        std::string approx = Rf_isNull(approximate_sexp) ? "none" :
                             std::string(CHAR(STRING_ELT(approximate_sexp, 0)));
        return at::gelu(self, approx);
}

// [[Rcpp::export]]
at::Tensor C_nnf_leaky_relu(at::Tensor self, SEXP negative_slope_sexp) {
        double ns = Rf_isNull(negative_slope_sexp) ? 0.01 : Rf_asReal(negative_slope_sexp);
        return at::leaky_relu(self, ns);
}

// [[Rcpp::export]]
at::Tensor C_nnf_elu(at::Tensor self, SEXP alpha_sexp) {
        double alpha = Rf_isNull(alpha_sexp) ? 1.0 : Rf_asReal(alpha_sexp);
        return at::elu(self, alpha);
}

// [[Rcpp::export]]
at::Tensor C_nnf_softmax(at::Tensor self, SEXP dim_sexp) {
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return at::softmax(self, dim);
}

// [[Rcpp::export]]
at::Tensor C_nnf_log_softmax(at::Tensor self, SEXP dim_sexp) {
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return at::log_softmax(self, dim);
}

// [[Rcpp::export]]
at::Tensor C_nnf_layer_norm(at::Tensor self, SEXP normalized_shape_sexp,
                                  SEXP weight, SEXP bias, SEXP eps_sexp) {
        auto nshape = sexp_to_int_vec(normalized_shape_sexp);
        double eps = Rf_isNull(eps_sexp) ? 1e-5 : Rf_asReal(eps_sexp);

        c10::optional<at::Tensor> w_opt;
        c10::optional<at::Tensor> b_opt;
        if (!Rf_isNull(weight)) {
            w_opt = *get_tensor_ptr(weight);
        }
        if (!Rf_isNull(bias)) {
            b_opt = *get_tensor_ptr(bias);
        }

        return at::layer_norm(self, at::IntArrayRef(nshape.data(), nshape.size()),
                           w_opt, b_opt, eps);
}

// ---- Linear algebra functions ----

// [[Rcpp::export]]
at::Tensor C_torch_linear(at::Tensor input, at::Tensor weight, SEXP bias) {
        if (Rf_isNull(bias)) {
            return at::linear(input, weight);
        } else {
            return at::linear(input, weight, *get_tensor_ptr(bias));
        }
}

// [[Rcpp::export]]
at::Tensor C_torch_conv1d(at::Tensor input, at::Tensor weight, SEXP bias,
                                SEXP stride_sexp, SEXP padding_sexp,
                                SEXP dilation_sexp, SEXP groups_sexp) {
        int64_t stride = static_cast<int64_t>(Rf_asInteger(stride_sexp));
        int64_t padding = static_cast<int64_t>(Rf_asInteger(padding_sexp));
        int64_t dilation = static_cast<int64_t>(Rf_asInteger(dilation_sexp));
        int64_t groups = static_cast<int64_t>(Rf_asInteger(groups_sexp));

        if (Rf_isNull(bias)) {
            return at::conv1d(input, weight, {}, stride, padding, dilation, groups);
        } else {
            return at::conv1d(input, weight, *get_tensor_ptr(bias),
                           stride, padding, dilation, groups);
        }
}

// [[Rcpp::export]]
at::Tensor C_torch_embedding(at::Tensor weight, at::Tensor indices) {
        // Convert 1-indexed R indices to 0-indexed ATen indices
        return at::embedding(weight, indices.sub(1));
}

// ---- Transposed convolution ----

// [[Rcpp::export]]
at::Tensor C_torch_conv_transpose1d(at::Tensor input, at::Tensor weight, SEXP bias,
                                          SEXP stride_sexp, SEXP padding_sexp,
                                          SEXP output_padding_sexp,
                                          SEXP groups_sexp, SEXP dilation_sexp) {
        int64_t stride = static_cast<int64_t>(Rf_asInteger(stride_sexp));
        int64_t padding = static_cast<int64_t>(Rf_asInteger(padding_sexp));
        int64_t output_padding = static_cast<int64_t>(Rf_asInteger(output_padding_sexp));
        int64_t groups = static_cast<int64_t>(Rf_asInteger(groups_sexp));
        int64_t dilation = static_cast<int64_t>(Rf_asInteger(dilation_sexp));

        if (Rf_isNull(bias)) {
            return at::conv_transpose1d(input, weight, {}, stride, padding,
                                     output_padding, groups, dilation);
        } else {
            return at::conv_transpose1d(input, weight, *get_tensor_ptr(bias),
                                     stride, padding, output_padding, groups, dilation);
        }
}

// ---- 2D convolution ----

// [[Rcpp::export]]
at::Tensor C_torch_conv2d(at::Tensor input, at::Tensor weight, SEXP bias,
                                SEXP stride_sexp, SEXP padding_sexp,
                                SEXP dilation_sexp, SEXP groups_sexp) {
        auto stride = sexp_to_int_vec(stride_sexp);
        auto padding = sexp_to_int_vec(padding_sexp);
        auto dilation = sexp_to_int_vec(dilation_sexp);
        int64_t groups = static_cast<int64_t>(Rf_asInteger(groups_sexp));

        if (Rf_isNull(bias)) {
            return at::conv2d(input, weight, {},
                           at::IntArrayRef(stride.data(), stride.size()),
                           at::IntArrayRef(padding.data(), padding.size()),
                           at::IntArrayRef(dilation.data(), dilation.size()),
                           groups);
        } else {
            return at::conv2d(input, weight, *get_tensor_ptr(bias),
                           at::IntArrayRef(stride.data(), stride.size()),
                           at::IntArrayRef(padding.data(), padding.size()),
                           at::IntArrayRef(dilation.data(), dilation.size()),
                           groups);
        }
}

// ---- Batch normalization ----

// [[Rcpp::export]]
at::Tensor C_torch_batch_norm(at::Tensor input, SEXP weight, SEXP bias,
                                    SEXP running_mean, SEXP running_var,
                                    SEXP training_sexp, SEXP momentum_sexp,
                                    SEXP eps_sexp, SEXP cudnn_enabled_sexp) {
        bool training = Rf_asLogical(training_sexp);
        double momentum = Rf_asReal(momentum_sexp);
        double eps = Rf_asReal(eps_sexp);
        bool cudnn_enabled = Rf_asLogical(cudnn_enabled_sexp);

        c10::optional<at::Tensor> w_opt, b_opt, rm_opt, rv_opt;
        if (!Rf_isNull(weight)) w_opt = *get_tensor_ptr(weight);
        if (!Rf_isNull(bias)) b_opt = *get_tensor_ptr(bias);
        if (!Rf_isNull(running_mean)) rm_opt = *get_tensor_ptr(running_mean);
        if (!Rf_isNull(running_var)) rv_opt = *get_tensor_ptr(running_var);

        return at::batch_norm(input, w_opt, b_opt, rm_opt, rv_opt,
                           training, momentum, eps, cudnn_enabled);
}

// ---- LSTM ----

// [[Rcpp::export]]
SEXP C_torch_lstm(SEXP input, SEXP hx_sexp, SEXP params_sexp,
                              SEXP has_biases_sexp, SEXP num_layers_sexp,
                              SEXP dropout_sexp, SEXP batch_first_sexp,
                              SEXP bidirectional_sexp) {
        auto* inp = get_tensor_ptr(input);
        bool has_biases = Rf_asLogical(has_biases_sexp);
        int64_t num_layers = static_cast<int64_t>(Rf_asInteger(num_layers_sexp));
        double dropout = Rf_asReal(dropout_sexp);
        bool batch_first = Rf_asLogical(batch_first_sexp);
        bool bidirectional = Rf_asLogical(bidirectional_sexp);
        bool training = false;  // Always inference mode

        // Collect parameters from R list
        R_xlen_t nparams = Rf_xlength(params_sexp);
        std::vector<at::Tensor> params;
        params.reserve(nparams);
        for (R_xlen_t i = 0; i < nparams; i++) {
            SEXP elem = VECTOR_ELT(params_sexp, i);
            if (!Rf_isNull(elem)) {
                params.push_back(*get_tensor_ptr(elem));
            }
        }

        // Handle initial hidden state
        int64_t num_directions = bidirectional ? 2 : 1;
        int64_t batch_size = batch_first ? inp->size(0) : inp->size(1);
        int64_t hidden_size = params[1].size(1);  // weight_hh has shape [4*hidden, hidden]

        at::Tensor h0, c0;
        if (Rf_isNull(hx_sexp)) {
            auto opts = inp->options();
            h0 = at::zeros({num_layers * num_directions, batch_size, hidden_size}, opts);
            c0 = at::zeros({num_layers * num_directions, batch_size, hidden_size}, opts);
        } else {
            // hx is a list of (h0, c0)
            h0 = *get_tensor_ptr(VECTOR_ELT(hx_sexp, 0));
            c0 = *get_tensor_ptr(VECTOR_ELT(hx_sexp, 1));
        }

        auto result = at::lstm(*inp, {h0, c0}, params, has_biases,
                               num_layers, dropout, training, bidirectional,
                               batch_first);

        // Return list(output, list(h_n, c_n)) to match torch R package
        SEXP hidden = PROTECT(Rf_allocVector(VECSXP, 2));
        SET_VECTOR_ELT(hidden, 0, make_tensor_sexp(new at::Tensor(std::get<1>(result))));
        SET_VECTOR_ELT(hidden, 1, make_tensor_sexp(new at::Tensor(std::get<2>(result))));

        SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));
        SET_VECTOR_ELT(out, 0, make_tensor_sexp(new at::Tensor(std::get<0>(result))));
        SET_VECTOR_ELT(out, 1, hidden);
        UNPROTECT(2);
        return out;
}

// ---- NN functional: pad ----

// [[Rcpp::export]]
at::Tensor C_nnf_pad(at::Tensor input, SEXP pad_sexp, SEXP mode_sexp,
                           SEXP value_sexp) {
        auto pad = sexp_to_int_vec(pad_sexp);
        double value = Rf_asReal(value_sexp);
        const char* mode = CHAR(STRING_ELT(mode_sexp, 0));

        std::string mode_str(mode);
        if (mode_str == "constant") {
            return at::constant_pad_nd(input,
                at::IntArrayRef(pad.data(), pad.size()), value);
        } else if (mode_str == "reflect") {
            return at::reflection_pad1d(input,
                at::IntArrayRef(pad.data(), pad.size()));
        } else if (mode_str == "replicate") {
            return at::replication_pad1d(input,
                at::IntArrayRef(pad.data(), pad.size()));
        } else {
            Rf_error("Unsupported padding mode: %s", mode);
        }
}

// ---- NN functional: interpolate ----

// [[Rcpp::export]]
SEXP C_nnf_interpolate(SEXP input, SEXP size_sexp,
                                    SEXP scale_factor_sexp, SEXP mode_sexp,
                                    SEXP align_corners_sexp) {
        auto* inp = get_tensor_ptr(input);
        const char* mode = CHAR(STRING_ELT(mode_sexp, 0));
        std::string mode_str(mode);

        int64_t ndim = inp->dim();
        // ndim: 3 = 1D, 4 = 2D, 5 = 3D
        // For 1D (3D tensor): [N, C, L]

        if (!Rf_isNull(size_sexp)) {
            auto size = sexp_to_int_vec(size_sexp);
            if (mode_str == "nearest") {
                if (ndim == 3) {
                    return make_tensor_sexp(new at::Tensor(
                        at::upsample_nearest1d(*inp, size)));
                } else if (ndim == 4) {
                    return make_tensor_sexp(new at::Tensor(
                        at::upsample_nearest2d(*inp, size)));
                }
            } else if (mode_str == "linear") {
                bool ac = !Rf_isNull(align_corners_sexp) && Rf_asLogical(align_corners_sexp);
                return make_tensor_sexp(new at::Tensor(
                    at::upsample_linear1d(*inp, size, ac)));
            } else if (mode_str == "bilinear") {
                bool ac = !Rf_isNull(align_corners_sexp) && Rf_asLogical(align_corners_sexp);
                return make_tensor_sexp(new at::Tensor(
                    at::upsample_bilinear2d(*inp, size, ac)));
            }
        } else if (!Rf_isNull(scale_factor_sexp)) {
            R_xlen_t nscale = Rf_xlength(scale_factor_sexp);
            std::vector<double> scales(nscale);
            for (R_xlen_t i = 0; i < nscale; i++) {
                scales[i] = REAL(scale_factor_sexp)[i];
            }

            if (mode_str == "nearest") {
                if (ndim == 3) {
                    return make_tensor_sexp(new at::Tensor(
                        at::upsample_nearest1d(*inp, c10::nullopt, scales[0])));
                } else if (ndim == 4) {
                    std::vector<double> sv = {scales[0], nscale > 1 ? scales[1] : scales[0]};
                    return make_tensor_sexp(new at::Tensor(
                        at::upsample_nearest2d(*inp, c10::nullopt,
                                               at::ArrayRef<double>(sv))));
                }
            }
        }

        Rf_error("Unsupported interpolate mode/input combination: mode=%s, ndim=%lld",
                 mode, (long long)ndim);
}

// ---- NN functional: avg_pool1d ----

// [[Rcpp::export]]
at::Tensor C_nnf_avg_pool1d(at::Tensor input, SEXP kernel_size_sexp,
                                   SEXP stride_sexp, SEXP padding_sexp,
                                   SEXP ceil_mode_sexp,
                                   SEXP count_include_pad_sexp) {
        int64_t kernel_size = static_cast<int64_t>(Rf_asInteger(kernel_size_sexp));
        int64_t stride = static_cast<int64_t>(Rf_asInteger(stride_sexp));
        int64_t padding = static_cast<int64_t>(Rf_asInteger(padding_sexp));
        bool ceil_mode = Rf_asLogical(ceil_mode_sexp);
        bool count_include_pad = Rf_asLogical(count_include_pad_sexp);

        return at::avg_pool1d(input, {kernel_size}, {stride}, {padding},
                           ceil_mode, count_include_pad);
}

// ---- NN functional: softplus ----

// [[Rcpp::export]]
at::Tensor C_nnf_softplus(at::Tensor input, SEXP beta_sexp, SEXP threshold_sexp) {
        double beta = Rf_asReal(beta_sexp);
        double threshold = Rf_asReal(threshold_sexp);
        return at::softplus(input, beta, threshold);
}

// ---- NN functional: normalize ----

// [[Rcpp::export]]
at::Tensor C_nnf_normalize(at::Tensor input, SEXP p_sexp, SEXP dim_sexp,
                                  SEXP eps_sexp) {
        double p = Rf_asReal(p_sexp);
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        double eps = Rf_asReal(eps_sexp);
        // F.normalize = x / max(norm(x, p, dim, keepdim=True), eps)
        auto norm = input.norm(p, dim, /*keepdim=*/true);
        auto clamped = at::clamp_min(norm, eps);
        return input.div(clamped);
}

// ---- Scaled Dot-Product Attention ----

// [[Rcpp::export]]
at::Tensor C_torch_sdpa(at::Tensor query, at::Tensor key, at::Tensor value,
                              SEXP attn_mask_sexp, SEXP dropout_sexp,
                              SEXP is_causal_sexp) {
        double dropout_p = Rf_asReal(dropout_sexp);
        bool is_causal = Rf_asLogical(is_causal_sexp);

        c10::optional<at::Tensor> attn_mask;
        if (!Rf_isNull(attn_mask_sexp) && TYPEOF(attn_mask_sexp) == EXTPTRSXP) {
            attn_mask = *get_tensor_ptr(attn_mask_sexp);
        }

        return at::scaled_dot_product_attention(
            query, key, value, attn_mask, dropout_p, is_causal);
}
