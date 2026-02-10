#include "Rtorch.h"

// ---- Activation functions ----

extern "C" SEXP C_nnf_silu(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(at::silu(*a)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_nnf_gelu(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(at::gelu(*a)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_nnf_leaky_relu(SEXP self, SEXP negative_slope_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        double ns = Rf_isNull(negative_slope_sexp) ? 0.01 : Rf_asReal(negative_slope_sexp);
        return make_tensor_sexp(new at::Tensor(at::leaky_relu(*a, ns)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_nnf_elu(SEXP self, SEXP alpha_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        double alpha = Rf_isNull(alpha_sexp) ? 1.0 : Rf_asReal(alpha_sexp);
        return make_tensor_sexp(new at::Tensor(at::elu(*a, alpha)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_nnf_softmax(SEXP self, SEXP dim_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return make_tensor_sexp(new at::Tensor(at::softmax(*a, dim)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_nnf_log_softmax(SEXP self, SEXP dim_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return make_tensor_sexp(new at::Tensor(at::log_softmax(*a, dim)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_nnf_layer_norm(SEXP self, SEXP normalized_shape_sexp,
                                  SEXP weight, SEXP bias, SEXP eps_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
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

        return make_tensor_sexp(new at::Tensor(
            at::layer_norm(*a, at::IntArrayRef(nshape.data(), nshape.size()),
                           w_opt, b_opt, eps)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

// ---- Linear algebra functions ----

extern "C" SEXP C_torch_linear(SEXP input, SEXP weight, SEXP bias) {
    try {
        auto* inp = get_tensor_ptr(input);
        auto* w = get_tensor_ptr(weight);

        if (Rf_isNull(bias)) {
            return make_tensor_sexp(new at::Tensor(at::linear(*inp, *w)));
        } else {
            auto* b = get_tensor_ptr(bias);
            return make_tensor_sexp(new at::Tensor(at::linear(*inp, *w, *b)));
        }
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_conv1d(SEXP input, SEXP weight, SEXP bias,
                                SEXP stride_sexp, SEXP padding_sexp,
                                SEXP dilation_sexp, SEXP groups_sexp) {
    try {
        auto* inp = get_tensor_ptr(input);
        auto* w = get_tensor_ptr(weight);

        int64_t stride = static_cast<int64_t>(Rf_asInteger(stride_sexp));
        int64_t padding = static_cast<int64_t>(Rf_asInteger(padding_sexp));
        int64_t dilation = static_cast<int64_t>(Rf_asInteger(dilation_sexp));
        int64_t groups = static_cast<int64_t>(Rf_asInteger(groups_sexp));

        if (Rf_isNull(bias)) {
            return make_tensor_sexp(new at::Tensor(
                at::conv1d(*inp, *w, {}, stride, padding, dilation, groups)));
        } else {
            auto* b = get_tensor_ptr(bias);
            return make_tensor_sexp(new at::Tensor(
                at::conv1d(*inp, *w, *b, stride, padding, dilation, groups)));
        }
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_embedding(SEXP weight, SEXP indices) {
    try {
        auto* w = get_tensor_ptr(weight);
        auto* idx = get_tensor_ptr(indices);
        return make_tensor_sexp(new at::Tensor(at::embedding(*w, *idx)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}
