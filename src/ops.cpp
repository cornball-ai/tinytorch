#include "Rtorch.h"

// ---- Arithmetic ops ----

extern "C" SEXP C_torch_add(SEXP self, SEXP other, SEXP alpha) {
    try {
        auto* a = get_tensor_ptr(self);
        auto* b = get_tensor_ptr(other);
        at::Scalar alpha_s = Rf_isNull(alpha) ? at::Scalar(1) : sexp_to_scalar(alpha);
        return make_tensor_sexp(new at::Tensor(a->add(*b, alpha_s)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_sub(SEXP self, SEXP other, SEXP alpha) {
    try {
        auto* a = get_tensor_ptr(self);
        auto* b = get_tensor_ptr(other);
        at::Scalar alpha_s = Rf_isNull(alpha) ? at::Scalar(1) : sexp_to_scalar(alpha);
        return make_tensor_sexp(new at::Tensor(a->sub(*b, alpha_s)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_mul(SEXP self, SEXP other) {
    try {
        auto* a = get_tensor_ptr(self);
        auto* b = get_tensor_ptr(other);
        return make_tensor_sexp(new at::Tensor(a->mul(*b)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_div(SEXP self, SEXP other) {
    try {
        auto* a = get_tensor_ptr(self);
        auto* b = get_tensor_ptr(other);
        return make_tensor_sexp(new at::Tensor(a->div(*b)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_neg(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->neg()));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_logical_not(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->logical_not()));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

// ---- Scalar arithmetic (tensor op scalar) ----

extern "C" SEXP C_torch_add_scalar(SEXP self, SEXP scalar) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->add(sexp_to_scalar(scalar))));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_sub_scalar(SEXP self, SEXP scalar) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->sub(sexp_to_scalar(scalar))));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_mul_scalar(SEXP self, SEXP scalar) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->mul(sexp_to_scalar(scalar))));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_div_scalar(SEXP self, SEXP scalar) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->div(sexp_to_scalar(scalar))));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

// ---- Linear algebra ----

extern "C" SEXP C_torch_matmul(SEXP self, SEXP other) {
    try {
        auto* a = get_tensor_ptr(self);
        auto* b = get_tensor_ptr(other);
        return make_tensor_sexp(new at::Tensor(a->matmul(*b)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_mm(SEXP self, SEXP other) {
    try {
        auto* a = get_tensor_ptr(self);
        auto* b = get_tensor_ptr(other);
        return make_tensor_sexp(new at::Tensor(a->mm(*b)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_t(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->t()));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

// ---- Reduction ----

extern "C" SEXP C_torch_sum(SEXP self, SEXP dim_sexp, SEXP keepdim_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        bool keepdim = Rf_asLogical(keepdim_sexp);

        if (Rf_isNull(dim_sexp)) {
            return make_tensor_sexp(new at::Tensor(a->sum()));
        }

        // 1-based to 0-based conversion
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return make_tensor_sexp(new at::Tensor(a->sum(dim, keepdim)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_mean(SEXP self, SEXP dim_sexp, SEXP keepdim_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        bool keepdim = Rf_asLogical(keepdim_sexp);

        if (Rf_isNull(dim_sexp)) {
            return make_tensor_sexp(new at::Tensor(a->mean()));
        }

        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return make_tensor_sexp(new at::Tensor(a->mean(dim, keepdim)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_max(SEXP self, SEXP dim_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        if (Rf_isNull(dim_sexp)) {
            return make_tensor_sexp(new at::Tensor(a->max()));
        }
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        auto result = a->max(dim);
        return make_tensor_sexp(new at::Tensor(std::get<0>(result)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_min(SEXP self, SEXP dim_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        if (Rf_isNull(dim_sexp)) {
            return make_tensor_sexp(new at::Tensor(a->min()));
        }
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        auto result = a->min(dim);
        return make_tensor_sexp(new at::Tensor(std::get<0>(result)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

// ---- Shape operations ----

extern "C" SEXP C_torch_reshape(SEXP self, SEXP shape_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        auto shape = sexp_to_int_vec(shape_sexp);
        return make_tensor_sexp(new at::Tensor(a->reshape(at::IntArrayRef(shape.data(), shape.size()))));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_view(SEXP self, SEXP shape_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        auto shape = sexp_to_int_vec(shape_sexp);
        return make_tensor_sexp(new at::Tensor(a->view(at::IntArrayRef(shape.data(), shape.size()))));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_squeeze(SEXP self, SEXP dim_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        if (Rf_isNull(dim_sexp)) {
            return make_tensor_sexp(new at::Tensor(a->squeeze()));
        }
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return make_tensor_sexp(new at::Tensor(a->squeeze(dim)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_unsqueeze(SEXP self, SEXP dim_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return make_tensor_sexp(new at::Tensor(a->unsqueeze(dim)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

// ---- Unary ops ----

extern "C" SEXP C_torch_relu(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->relu()));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_sigmoid(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->sigmoid()));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_tanh(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->tanh()));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_exp(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->exp()));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_log(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->log()));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_log2(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->log2()));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_log10(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->log10()));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_sqrt(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->sqrt()));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_abs(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->abs()));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_sign(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->sign()));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_floor(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->floor()));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_ceil(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->ceil()));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_round(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->round()));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_trunc(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->trunc()));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_sin(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->sin()));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_cos(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->cos()));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_rsqrt(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->rsqrt()));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_detach(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->detach()));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

// ---- Binary ops ----

extern "C" SEXP C_torch_pow(SEXP self, SEXP other) {
    try {
        auto* a = get_tensor_ptr(self);
        auto* b = get_tensor_ptr(other);
        return make_tensor_sexp(new at::Tensor(a->pow(*b)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_pow_scalar(SEXP self, SEXP scalar) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->pow(sexp_to_scalar(scalar))));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

// scalar ^ tensor
extern "C" SEXP C_torch_scalar_pow(SEXP scalar, SEXP exponent) {
    try {
        auto* b = get_tensor_ptr(exponent);
        return make_tensor_sexp(new at::Tensor(at::pow(sexp_to_scalar(scalar), *b)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_remainder(SEXP self, SEXP other) {
    try {
        auto* a = get_tensor_ptr(self);
        auto* b = get_tensor_ptr(other);
        return make_tensor_sexp(new at::Tensor(a->remainder(*b)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_remainder_scalar(SEXP self, SEXP scalar) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->remainder(sexp_to_scalar(scalar))));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_floor_divide(SEXP self, SEXP other) {
    try {
        auto* a = get_tensor_ptr(self);
        auto* b = get_tensor_ptr(other);
        return make_tensor_sexp(new at::Tensor(a->floor_divide(*b)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_floor_divide_scalar(SEXP self, SEXP scalar) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->floor_divide(sexp_to_scalar(scalar))));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

// ---- Comparison ops ----

extern "C" SEXP C_torch_eq(SEXP self, SEXP other) {
    try {
        auto* a = get_tensor_ptr(self);
        auto* b = get_tensor_ptr(other);
        return make_tensor_sexp(new at::Tensor(a->eq(*b)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_eq_scalar(SEXP self, SEXP scalar) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->eq(sexp_to_scalar(scalar))));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_ne(SEXP self, SEXP other) {
    try {
        auto* a = get_tensor_ptr(self);
        auto* b = get_tensor_ptr(other);
        return make_tensor_sexp(new at::Tensor(a->ne(*b)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_ne_scalar(SEXP self, SEXP scalar) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->ne(sexp_to_scalar(scalar))));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_lt(SEXP self, SEXP other) {
    try {
        auto* a = get_tensor_ptr(self);
        auto* b = get_tensor_ptr(other);
        return make_tensor_sexp(new at::Tensor(a->lt(*b)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_lt_scalar(SEXP self, SEXP scalar) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->lt(sexp_to_scalar(scalar))));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_le(SEXP self, SEXP other) {
    try {
        auto* a = get_tensor_ptr(self);
        auto* b = get_tensor_ptr(other);
        return make_tensor_sexp(new at::Tensor(a->le(*b)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_le_scalar(SEXP self, SEXP scalar) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->le(sexp_to_scalar(scalar))));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_gt(SEXP self, SEXP other) {
    try {
        auto* a = get_tensor_ptr(self);
        auto* b = get_tensor_ptr(other);
        return make_tensor_sexp(new at::Tensor(a->gt(*b)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_gt_scalar(SEXP self, SEXP scalar) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->gt(sexp_to_scalar(scalar))));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_ge(SEXP self, SEXP other) {
    try {
        auto* a = get_tensor_ptr(self);
        auto* b = get_tensor_ptr(other);
        return make_tensor_sexp(new at::Tensor(a->ge(*b)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_ge_scalar(SEXP self, SEXP scalar) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->ge(sexp_to_scalar(scalar))));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

// ---- Additional linear algebra / shape ----

extern "C" SEXP C_torch_bmm(SEXP self, SEXP other) {
    try {
        auto* a = get_tensor_ptr(self);
        auto* b = get_tensor_ptr(other);
        return make_tensor_sexp(new at::Tensor(a->bmm(*b)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_transpose(SEXP self, SEXP dim0_sexp, SEXP dim1_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        int64_t dim0 = static_cast<int64_t>(Rf_asInteger(dim0_sexp));
        int64_t dim1 = static_cast<int64_t>(Rf_asInteger(dim1_sexp));
        if (dim0 > 0) dim0 = dim0 - 1;
        if (dim1 > 0) dim1 = dim1 - 1;
        return make_tensor_sexp(new at::Tensor(a->transpose(dim0, dim1)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_flatten(SEXP self, SEXP start_dim_sexp, SEXP end_dim_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        int64_t start_dim = static_cast<int64_t>(Rf_asInteger(start_dim_sexp));
        int64_t end_dim = static_cast<int64_t>(Rf_asInteger(end_dim_sexp));
        if (start_dim > 0) start_dim = start_dim - 1;
        if (end_dim > 0) end_dim = end_dim - 1;
        return make_tensor_sexp(new at::Tensor(a->flatten(start_dim, end_dim)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

// ---- Utility ----

extern "C" SEXP C_torch_clone(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->clone()));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_contiguous(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->contiguous()));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_to_dtype(SEXP self, SEXP dtype_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        auto dtype = sexp_to_dtype(dtype_sexp);
        if (!dtype.has_value()) {
            Rf_error("dtype must be specified");
        }
        return make_tensor_sexp(new at::Tensor(a->to(dtype.value())));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_item(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        if (a->numel() != 1) {
            Rf_error("item() requires a tensor with exactly one element");
        }
        double val = a->item<double>();
        return Rf_ScalarReal(val);
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

// ---- New tensor operations ----

extern "C" SEXP C_torch_cat(SEXP tensors_sexp, SEXP dim_sexp) {
    try {
        R_xlen_t n = Rf_xlength(tensors_sexp);
        std::vector<at::Tensor> tensors;
        tensors.reserve(n);
        for (R_xlen_t i = 0; i < n; i++) {
            tensors.push_back(*get_tensor_ptr(VECTOR_ELT(tensors_sexp, i)));
        }
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return make_tensor_sexp(new at::Tensor(at::cat(tensors, dim)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_clamp(SEXP self, SEXP min_sexp, SEXP max_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        c10::optional<at::Scalar> min_val;
        c10::optional<at::Scalar> max_val;
        if (!Rf_isNull(min_sexp)) min_val = at::Scalar(Rf_asReal(min_sexp));
        if (!Rf_isNull(max_sexp)) max_val = at::Scalar(Rf_asReal(max_sexp));
        return make_tensor_sexp(new at::Tensor(at::clamp(*a, min_val, max_val)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_where(SEXP condition, SEXP self, SEXP other) {
    try {
        auto* cond = get_tensor_ptr(condition);
        auto* a = get_tensor_ptr(self);
        auto* b = get_tensor_ptr(other);
        return make_tensor_sexp(new at::Tensor(at::where(*cond, *a, *b)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_sort(SEXP self, SEXP dim_sexp, SEXP descending_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        bool descending = Rf_asLogical(descending_sexp);
        auto result = a->sort(dim, descending);
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
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_flip(SEXP self, SEXP dims_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        auto dims = sexp_to_int_vec(dims_sexp);
        // Convert 1-based to 0-based
        for (auto& d : dims) {
            if (d > 0) d = d - 1;
        }
        return make_tensor_sexp(new at::Tensor(
            a->flip(at::IntArrayRef(dims.data(), dims.size()))));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_cumsum(SEXP self, SEXP dim_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return make_tensor_sexp(new at::Tensor(a->cumsum(dim)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_maximum(SEXP self, SEXP other) {
    try {
        auto* a = get_tensor_ptr(self);
        auto* b = get_tensor_ptr(other);
        return make_tensor_sexp(new at::Tensor(at::maximum(*a, *b)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_multinomial(SEXP self, SEXP num_samples_sexp,
                                     SEXP replacement_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        int64_t num_samples = static_cast<int64_t>(Rf_asInteger(num_samples_sexp));
        bool replacement = Rf_asLogical(replacement_sexp);
        // Convert to 1-indexed (R convention, matching torch R package)
        return make_tensor_sexp(new at::Tensor(
            a->multinomial(num_samples, replacement).add(1)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_outer(SEXP self, SEXP vec2) {
    try {
        auto* a = get_tensor_ptr(self);
        auto* b = get_tensor_ptr(vec2);
        return make_tensor_sexp(new at::Tensor(at::outer(*a, *b)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_triu(SEXP self, SEXP diagonal_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        int64_t diagonal = static_cast<int64_t>(Rf_asInteger(diagonal_sexp));
        return make_tensor_sexp(new at::Tensor(a->triu(diagonal)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_norm(SEXP self, SEXP p_sexp, SEXP dim_sexp,
                              SEXP keepdim_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        double p = Rf_asReal(p_sexp);
        bool keepdim = Rf_asLogical(keepdim_sexp);
        if (Rf_isNull(dim_sexp)) {
            return make_tensor_sexp(new at::Tensor(a->norm(p)));
        }
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return make_tensor_sexp(new at::Tensor(a->norm(p, dim, keepdim)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_std(SEXP self, SEXP dim_sexp, SEXP keepdim_sexp,
                            SEXP correction_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        bool keepdim = Rf_asLogical(keepdim_sexp);
        bool unbiased = Rf_isNull(correction_sexp) ? true :
                        (Rf_asLogical(correction_sexp) != 0);
        if (Rf_isNull(dim_sexp)) {
            return make_tensor_sexp(new at::Tensor(a->std(unbiased)));
        }
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return make_tensor_sexp(new at::Tensor(
            a->std({dim}, unbiased, keepdim)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

// ---- Complex & signal processing ----

extern "C" SEXP C_torch_complex(SEXP real, SEXP imag) {
    try {
        auto* r = get_tensor_ptr(real);
        auto* i = get_tensor_ptr(imag);
        return make_tensor_sexp(new at::Tensor(at::complex(*r, *i)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_real(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(at::real(*a)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_imag(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(at::imag(*a)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_polar(SEXP abs, SEXP angle) {
    try {
        auto* a = get_tensor_ptr(abs);
        auto* b = get_tensor_ptr(angle);
        return make_tensor_sexp(new at::Tensor(at::polar(*a, *b)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_view_as_real(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(at::view_as_real(*a)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_stft(SEXP input, SEXP n_fft_sexp, SEXP hop_sexp,
                              SEXP win_length_sexp, SEXP window,
                              SEXP center_sexp, SEXP normalized_sexp,
                              SEXP onesided_sexp, SEXP return_complex_sexp) {
    try {
        auto* inp = get_tensor_ptr(input);
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
        at::Tensor x = *inp;
        if (center) {
            int64_t pad_amount = n_fft / 2;
            // reflection_pad1d expects 3D input (batch, channel, length)
            bool needs_unsqueeze = (x.dim() == 1);
            if (needs_unsqueeze) x = x.unsqueeze(0).unsqueeze(0);
            else if (x.dim() == 2) x = x.unsqueeze(1);
            x = at::reflection_pad1d(x, {pad_amount, pad_amount});
            if (needs_unsqueeze) x = x.squeeze(0).squeeze(0);
            else if (inp->dim() == 2) x = x.squeeze(1);
        }

        return make_tensor_sexp(new at::Tensor(
            at::stft(x, n_fft, hop, win_length, win_opt,
                     normalized, onesided, return_complex)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_istft(SEXP input, SEXP n_fft_sexp, SEXP hop_sexp,
                               SEXP win_length_sexp, SEXP window,
                               SEXP center_sexp, SEXP normalized_sexp,
                               SEXP onesided_sexp, SEXP length_sexp,
                               SEXP return_complex_sexp) {
    try {
        auto* inp = get_tensor_ptr(input);
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

        return make_tensor_sexp(new at::Tensor(
            at::istft(*inp, n_fft, hop, win_length, win_opt,
                      center, normalized, onesided, len_opt, return_complex)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_hann_window(SEXP length_sexp, SEXP periodic_sexp,
                                     SEXP dtype_sexp) {
    try {
        int64_t length = static_cast<int64_t>(Rf_asInteger(length_sexp));
        bool periodic = Rf_asLogical(periodic_sexp);
        auto opts = at::TensorOptions();
        auto dtype = sexp_to_dtype(dtype_sexp);
        if (dtype.has_value()) opts = opts.dtype(dtype.value());
        return make_tensor_sexp(new at::Tensor(
            torch::hann_window(length, periodic, opts)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}
