#include "Rtorch.h"

// ---- Arithmetic ops ----

// [[Rcpp::export]]
at::Tensor C_torch_add(at::Tensor self, at::Tensor other, SEXP alpha) {
        at::Scalar alpha_s = Rf_isNull(alpha) ? at::Scalar(1) : sexp_to_scalar(alpha);
        return self.add(other, alpha_s);
}

// [[Rcpp::export]]
at::Tensor C_torch_sub(at::Tensor self, at::Tensor other, SEXP alpha) {
        at::Scalar alpha_s = Rf_isNull(alpha) ? at::Scalar(1) : sexp_to_scalar(alpha);
        return self.sub(other, alpha_s);
}

// [[Rcpp::export]]
at::Tensor C_torch_mul(at::Tensor self, at::Tensor other) { return self.mul(other); }

// [[Rcpp::export]]
at::Tensor C_torch_div(at::Tensor self, at::Tensor other) { return self.div(other); }

// [[Rcpp::export]]
at::Tensor C_torch_neg(at::Tensor self) { return self.neg(); }

// [[Rcpp::export]]
at::Tensor C_torch_logical_not(at::Tensor self) { return self.logical_not(); }

// ---- Scalar arithmetic (tensor op scalar) ----

// [[Rcpp::export]]
at::Tensor C_torch_add_scalar(at::Tensor self, SEXP scalar) {
        return self.add(sexp_to_scalar(scalar));
}

// [[Rcpp::export]]
at::Tensor C_torch_sub_scalar(at::Tensor self, SEXP scalar) {
        return self.sub(sexp_to_scalar(scalar));
}

// [[Rcpp::export]]
at::Tensor C_torch_mul_scalar(at::Tensor self, SEXP scalar) {
        return self.mul(sexp_to_scalar(scalar));
}

// [[Rcpp::export]]
at::Tensor C_torch_div_scalar(at::Tensor self, SEXP scalar) {
        return self.div(sexp_to_scalar(scalar));
}

// ---- Linear algebra ----

// [[Rcpp::export]]
at::Tensor C_torch_matmul(at::Tensor self, at::Tensor other) { return self.matmul(other); }

// [[Rcpp::export]]
at::Tensor C_torch_mm(at::Tensor self, at::Tensor other) { return self.mm(other); }

// [[Rcpp::export]]
at::Tensor C_torch_t(at::Tensor self) { return self.t(); }

// ---- Reduction ----

// [[Rcpp::export]]
at::Tensor C_torch_sum(at::Tensor self, SEXP dim_sexp, SEXP keepdim_sexp) {
        bool keepdim = Rf_asLogical(keepdim_sexp);
        if (Rf_isNull(dim_sexp)) {
            return self.sum();
        }
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return self.sum(dim, keepdim);
}

// [[Rcpp::export]]
at::Tensor C_torch_mean(at::Tensor self, SEXP dim_sexp, SEXP keepdim_sexp) {
        bool keepdim = Rf_asLogical(keepdim_sexp);
        if (Rf_isNull(dim_sexp)) {
            return self.mean();
        }
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return self.mean(dim, keepdim);
}

// [[Rcpp::export]]
at::Tensor C_torch_max(at::Tensor self, SEXP dim_sexp) {
        if (Rf_isNull(dim_sexp)) {
            return self.max();
        }
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return std::get<0>(self.max(dim));
}

// [[Rcpp::export]]
at::Tensor C_torch_min(at::Tensor self, SEXP dim_sexp) {
        if (Rf_isNull(dim_sexp)) {
            return self.min();
        }
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return std::get<0>(self.min(dim));
}

// [[Rcpp::export]]
at::Tensor C_torch_argmax(at::Tensor self, SEXP dim_sexp, SEXP keepdim_sexp) {
        bool keepdim = Rf_asLogical(keepdim_sexp);
        if (Rf_isNull(dim_sexp)) {
            return self.argmax();
        }
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return self.argmax(dim, keepdim);
}

// [[Rcpp::export]]
at::Tensor C_torch_argmin(at::Tensor self, SEXP dim_sexp, SEXP keepdim_sexp) {
        bool keepdim = Rf_asLogical(keepdim_sexp);
        if (Rf_isNull(dim_sexp)) {
            return self.argmin();
        }
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return self.argmin(dim, keepdim);
}

// ---- Shape operations ----

// [[Rcpp::export]]
at::Tensor C_torch_reshape(at::Tensor self, SEXP shape_sexp) {
        auto shape = sexp_to_int_vec(shape_sexp);
        return self.reshape(at::IntArrayRef(shape.data(), shape.size()));
}

// [[Rcpp::export]]
at::Tensor C_torch_view(at::Tensor self, SEXP shape_sexp) {
        auto shape = sexp_to_int_vec(shape_sexp);
        return self.view(at::IntArrayRef(shape.data(), shape.size()));
}

// [[Rcpp::export]]
at::Tensor C_torch_squeeze(at::Tensor self, SEXP dim_sexp) {
        if (Rf_isNull(dim_sexp)) {
            return self.squeeze();
        }
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return self.squeeze(dim);
}

// [[Rcpp::export]]
at::Tensor C_torch_unsqueeze(at::Tensor self, SEXP dim_sexp) {
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return self.unsqueeze(dim);
}

// ---- Unary ops ----

// [[Rcpp::export]]
at::Tensor C_torch_relu(at::Tensor self) { return self.relu(); }

// [[Rcpp::export]]
at::Tensor C_torch_sigmoid(at::Tensor self) { return self.sigmoid(); }

// [[Rcpp::export]]
at::Tensor C_torch_tanh(at::Tensor self) { return self.tanh(); }

// [[Rcpp::export]]
at::Tensor C_torch_exp(at::Tensor self) { return self.exp(); }

// [[Rcpp::export]]
at::Tensor C_torch_log(at::Tensor self) { return self.log(); }

// [[Rcpp::export]]
at::Tensor C_torch_log2(at::Tensor self) { return self.log2(); }

// [[Rcpp::export]]
at::Tensor C_torch_log10(at::Tensor self) { return self.log10(); }

// [[Rcpp::export]]
at::Tensor C_torch_sqrt(at::Tensor self) { return self.sqrt(); }

// [[Rcpp::export]]
at::Tensor C_torch_abs(at::Tensor self) { return self.abs(); }

// [[Rcpp::export]]
at::Tensor C_torch_sign(at::Tensor self) { return self.sign(); }

// [[Rcpp::export]]
at::Tensor C_torch_floor(at::Tensor self) { return self.floor(); }

// [[Rcpp::export]]
at::Tensor C_torch_ceil(at::Tensor self) { return self.ceil(); }

// [[Rcpp::export]]
at::Tensor C_torch_round(at::Tensor self) { return self.round(); }

// [[Rcpp::export]]
at::Tensor C_torch_trunc(at::Tensor self) { return self.trunc(); }

// [[Rcpp::export]]
at::Tensor C_torch_sin(at::Tensor self) { return self.sin(); }

// [[Rcpp::export]]
at::Tensor C_torch_cos(at::Tensor self) { return self.cos(); }

// [[Rcpp::export]]
at::Tensor C_torch_rsqrt(at::Tensor self) { return self.rsqrt(); }

// [[Rcpp::export]]
at::Tensor C_torch_detach(at::Tensor self) { return self.detach(); }

// ---- Binary ops ----

// [[Rcpp::export]]
at::Tensor C_torch_pow(at::Tensor self, at::Tensor other) { return self.pow(other); }

// [[Rcpp::export]]
at::Tensor C_torch_pow_scalar(at::Tensor self, SEXP scalar) {
        return self.pow(sexp_to_scalar(scalar));
}

// scalar ^ tensor
// [[Rcpp::export]]
at::Tensor C_torch_scalar_pow(SEXP scalar, at::Tensor exponent) {
        return at::pow(sexp_to_scalar(scalar), exponent);
}

// [[Rcpp::export]]
at::Tensor C_torch_remainder(at::Tensor self, at::Tensor other) { return self.remainder(other); }

// [[Rcpp::export]]
at::Tensor C_torch_remainder_scalar(at::Tensor self, SEXP scalar) {
        return self.remainder(sexp_to_scalar(scalar));
}

// [[Rcpp::export]]
at::Tensor C_torch_floor_divide(at::Tensor self, at::Tensor other) { return self.floor_divide(other); }

// [[Rcpp::export]]
at::Tensor C_torch_floor_divide_scalar(at::Tensor self, SEXP scalar) {
        return self.floor_divide(sexp_to_scalar(scalar));
}

// ---- Comparison ops ----

// [[Rcpp::export]]
at::Tensor C_torch_eq(at::Tensor self, at::Tensor other) { return self.eq(other); }

// [[Rcpp::export]]
at::Tensor C_torch_eq_scalar(at::Tensor self, SEXP scalar) {
        return self.eq(sexp_to_scalar(scalar));
}

// [[Rcpp::export]]
at::Tensor C_torch_ne(at::Tensor self, at::Tensor other) { return self.ne(other); }

// [[Rcpp::export]]
at::Tensor C_torch_ne_scalar(at::Tensor self, SEXP scalar) {
        return self.ne(sexp_to_scalar(scalar));
}

// [[Rcpp::export]]
at::Tensor C_torch_lt(at::Tensor self, at::Tensor other) { return self.lt(other); }

// [[Rcpp::export]]
at::Tensor C_torch_lt_scalar(at::Tensor self, SEXP scalar) {
        return self.lt(sexp_to_scalar(scalar));
}

// [[Rcpp::export]]
at::Tensor C_torch_le(at::Tensor self, at::Tensor other) { return self.le(other); }

// [[Rcpp::export]]
at::Tensor C_torch_le_scalar(at::Tensor self, SEXP scalar) {
        return self.le(sexp_to_scalar(scalar));
}

// [[Rcpp::export]]
at::Tensor C_torch_gt(at::Tensor self, at::Tensor other) { return self.gt(other); }

// [[Rcpp::export]]
at::Tensor C_torch_gt_scalar(at::Tensor self, SEXP scalar) {
        return self.gt(sexp_to_scalar(scalar));
}

// [[Rcpp::export]]
at::Tensor C_torch_ge(at::Tensor self, at::Tensor other) { return self.ge(other); }

// [[Rcpp::export]]
at::Tensor C_torch_ge_scalar(at::Tensor self, SEXP scalar) {
        return self.ge(sexp_to_scalar(scalar));
}

// ---- Additional linear algebra / shape ----

// [[Rcpp::export]]
at::Tensor C_torch_bmm(at::Tensor self, at::Tensor other) { return self.bmm(other); }

// [[Rcpp::export]]
at::Tensor C_torch_transpose(at::Tensor self, SEXP dim0_sexp, SEXP dim1_sexp) {
        int64_t dim0 = static_cast<int64_t>(Rf_asInteger(dim0_sexp));
        int64_t dim1 = static_cast<int64_t>(Rf_asInteger(dim1_sexp));
        if (dim0 > 0) dim0 = dim0 - 1;
        if (dim1 > 0) dim1 = dim1 - 1;
        return self.transpose(dim0, dim1);
}

// [[Rcpp::export]]
at::Tensor C_torch_flatten(at::Tensor self, SEXP start_dim_sexp, SEXP end_dim_sexp) {
        int64_t start_dim = static_cast<int64_t>(Rf_asInteger(start_dim_sexp));
        int64_t end_dim = static_cast<int64_t>(Rf_asInteger(end_dim_sexp));
        if (start_dim > 0) start_dim = start_dim - 1;
        if (end_dim > 0) end_dim = end_dim - 1;
        return self.flatten(start_dim, end_dim);
}

// ---- Utility ----

// [[Rcpp::export]]
at::Tensor C_torch_clone(at::Tensor self) { return self.clone(); }

// [[Rcpp::export]]
at::Tensor C_torch_contiguous(at::Tensor self) { return self.contiguous(); }

// [[Rcpp::export]]
at::Tensor C_torch_to_dtype(at::Tensor self, SEXP dtype_sexp) {
        auto dtype = sexp_to_dtype(dtype_sexp);
        if (!dtype.has_value()) {
            Rf_error("dtype must be specified");
        }
        return self.to(dtype.value());
}

// [[Rcpp::export]]
SEXP C_torch_item(at::Tensor self) {
        if (self.numel() != 1) {
            Rf_error("item() requires a tensor with exactly one element");
        }
        return Rf_ScalarReal(self.item<double>());
}

// ---- New tensor operations ----

// [[Rcpp::export]]
SEXP C_torch_cat(SEXP tensors_sexp, SEXP dim_sexp) {
        R_xlen_t n = Rf_xlength(tensors_sexp);
        std::vector<at::Tensor> tensors;
        tensors.reserve(n);
        for (R_xlen_t i = 0; i < n; i++) {
            tensors.push_back(*get_tensor_ptr(VECTOR_ELT(tensors_sexp, i)));
        }
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return make_tensor_sexp(new at::Tensor(at::cat(tensors, dim)));
}

// [[Rcpp::export]]
at::Tensor C_torch_clamp(at::Tensor self, SEXP min_sexp, SEXP max_sexp) {
        c10::optional<at::Scalar> min_val;
        c10::optional<at::Scalar> max_val;
        if (!Rf_isNull(min_sexp)) min_val = at::Scalar(Rf_asReal(min_sexp));
        if (!Rf_isNull(max_sexp)) max_val = at::Scalar(Rf_asReal(max_sexp));
        return at::clamp(self, min_val, max_val);
}

// [[Rcpp::export]]
at::Tensor C_torch_where(at::Tensor condition, at::Tensor self, at::Tensor other) {
        return at::where(condition, self, other);
}

// [[Rcpp::export]]
SEXP C_torch_sort(at::Tensor self, SEXP dim_sexp, SEXP descending_sexp) {
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        bool descending = Rf_asLogical(descending_sexp);
        auto result = self.sort(dim, descending);
        // Return as R list with $values and $indices
        // Convert indices to 1-indexed (R convention, matching torch R package)
        auto indices = std::get<1>(result).add(1);
        SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));
        SET_VECTOR_ELT(out, 0, make_tensor_sexp(new at::Tensor(std::get<0>(result))));
        SET_VECTOR_ELT(out, 1, make_tensor_sexp(new at::Tensor(indices)));
        SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));
        SET_STRING_ELT(names, 0, Rf_mkChar("values"));
        SET_STRING_ELT(names, 1, Rf_mkChar("indices"));
        Rf_setAttrib(out, R_NamesSymbol, names);
        UNPROTECT(2);
        return out;
}

// [[Rcpp::export]]
at::Tensor C_torch_flip(at::Tensor self, SEXP dims_sexp) {
        auto dims = sexp_to_int_vec(dims_sexp);
        // Convert 1-based to 0-based
        for (auto& d : dims) {
            if (d > 0) d = d - 1;
        }
        return self.flip(at::IntArrayRef(dims.data(), dims.size()));
}

// [[Rcpp::export]]
at::Tensor C_torch_cumsum(at::Tensor self, SEXP dim_sexp) {
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return self.cumsum(dim);
}

// [[Rcpp::export]]
at::Tensor C_torch_maximum(at::Tensor self, at::Tensor other) {
        return at::maximum(self, other);
}

// [[Rcpp::export]]
at::Tensor C_torch_multinomial(at::Tensor self, SEXP num_samples_sexp,
                                     SEXP replacement_sexp) {
        int64_t num_samples = static_cast<int64_t>(Rf_asInteger(num_samples_sexp));
        bool replacement = Rf_asLogical(replacement_sexp);
        // Convert to 1-indexed (R convention, matching torch R package)
        return self.multinomial(num_samples, replacement).add(1);
}

// [[Rcpp::export]]
at::Tensor C_torch_outer(at::Tensor self, at::Tensor vec2) {
        return at::outer(self, vec2);
}

// [[Rcpp::export]]
at::Tensor C_torch_triu(at::Tensor self, SEXP diagonal_sexp) {
        int64_t diagonal = static_cast<int64_t>(Rf_asInteger(diagonal_sexp));
        return self.triu(diagonal);
}

// [[Rcpp::export]]
at::Tensor C_torch_norm(at::Tensor self, SEXP p_sexp, SEXP dim_sexp,
                              SEXP keepdim_sexp) {
        double p = Rf_asReal(p_sexp);
        bool keepdim = Rf_asLogical(keepdim_sexp);
        if (Rf_isNull(dim_sexp)) {
            return self.norm(p);
        }
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return self.norm(p, dim, keepdim);
}

// [[Rcpp::export]]
at::Tensor C_torch_std(at::Tensor self, SEXP dim_sexp, SEXP keepdim_sexp,
                            SEXP correction_sexp) {
        bool keepdim = Rf_asLogical(keepdim_sexp);
        bool unbiased = Rf_isNull(correction_sexp) ? true :
                        (Rf_asLogical(correction_sexp) != 0);
        if (Rf_isNull(dim_sexp)) {
            return self.std(unbiased);
        }
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return self.std({dim}, unbiased, keepdim);
}

// ---- Complex & signal processing ----

// [[Rcpp::export]]
at::Tensor C_torch_complex(at::Tensor real, at::Tensor imag) {
        return at::complex(real, imag);
}

// [[Rcpp::export]]
at::Tensor C_torch_real(at::Tensor self) { return at::real(self); }

// [[Rcpp::export]]
at::Tensor C_torch_imag(at::Tensor self) { return at::imag(self); }

// [[Rcpp::export]]
at::Tensor C_torch_polar(at::Tensor abs, at::Tensor angle) {
        return at::polar(abs, angle);
}

// [[Rcpp::export]]
at::Tensor C_torch_view_as_real(at::Tensor self) { return at::view_as_real(self); }

// [[Rcpp::export]]
at::Tensor C_torch_stft(at::Tensor input, SEXP n_fft_sexp, SEXP hop_sexp,
                              SEXP win_length_sexp, SEXP window,
                              SEXP center_sexp, SEXP normalized_sexp,
                              SEXP onesided_sexp, SEXP return_complex_sexp) {
        int64_t n_fft = static_cast<int64_t>(Rf_asInteger(n_fft_sexp));
        int64_t hop = static_cast<int64_t>(Rf_asInteger(hop_sexp));
        int64_t win_length = static_cast<int64_t>(Rf_asInteger(win_length_sexp));
        bool center = Rf_asLogical(center_sexp);
        bool normalized = Rf_asLogical(normalized_sexp);
        bool onesided = Rf_asLogical(onesided_sexp);
        bool return_complex = Rf_asLogical(return_complex_sexp);

        c10::optional<at::Tensor> win_opt;
        if (!Rf_isNull(window)) {
            win_opt = *get_tensor_ptr(window);
        }

        // If center=TRUE, pad input with reflect padding (matching PyTorch default)
        at::Tensor x = input;
        if (center) {
            int64_t pad_amount = n_fft / 2;
            // reflection_pad1d expects 3D input (batch, channel, length)
            bool needs_unsqueeze = (x.dim() == 1);
            if (needs_unsqueeze) x = x.unsqueeze(0).unsqueeze(0);
            else if (x.dim() == 2) x = x.unsqueeze(1);
            x = at::reflection_pad1d(x, {pad_amount, pad_amount});
            if (needs_unsqueeze) x = x.squeeze(0).squeeze(0);
            else if (input.dim() == 2) x = x.squeeze(1);
        }

        return at::stft(x, n_fft, hop, win_length, win_opt,
                     normalized, onesided, return_complex);
}

// [[Rcpp::export]]
at::Tensor C_torch_istft(at::Tensor input, SEXP n_fft_sexp, SEXP hop_sexp,
                               SEXP win_length_sexp, SEXP window,
                               SEXP center_sexp, SEXP normalized_sexp,
                               SEXP onesided_sexp, SEXP length_sexp,
                               SEXP return_complex_sexp) {
        int64_t n_fft = static_cast<int64_t>(Rf_asInteger(n_fft_sexp));
        int64_t hop = static_cast<int64_t>(Rf_asInteger(hop_sexp));
        int64_t win_length = static_cast<int64_t>(Rf_asInteger(win_length_sexp));
        bool center = Rf_asLogical(center_sexp);
        bool normalized = Rf_asLogical(normalized_sexp);
        bool onesided = Rf_asLogical(onesided_sexp);
        bool return_complex = Rf_asLogical(return_complex_sexp);

        c10::optional<at::Tensor> win_opt;
        if (!Rf_isNull(window)) {
            win_opt = *get_tensor_ptr(window);
        }

        c10::optional<int64_t> len_opt;
        if (!Rf_isNull(length_sexp)) {
            len_opt = static_cast<int64_t>(Rf_asInteger(length_sexp));
        }

        return at::istft(input, n_fft, hop, win_length, win_opt,
                      center, normalized, onesided, len_opt, return_complex);
}

// [[Rcpp::export]]
SEXP C_torch_hann_window(SEXP length_sexp, SEXP periodic_sexp,
                                     SEXP dtype_sexp, SEXP device_sexp) {
        int64_t length = static_cast<int64_t>(Rf_asInteger(length_sexp));
        bool periodic = Rf_asLogical(periodic_sexp);
        auto opts = at::TensorOptions();
        auto dtype = sexp_to_dtype(dtype_sexp);
        if (dtype.has_value()) opts = opts.dtype(dtype.value());
        if (!Rf_isNull(device_sexp)) opts = opts.device(sexp_to_device(device_sexp));
        return make_tensor_sexp(new at::Tensor(
            torch::hann_window(length, periodic, opts)));
}
