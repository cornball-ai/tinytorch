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
