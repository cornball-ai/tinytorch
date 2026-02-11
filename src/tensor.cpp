#include "Rtorch.h"

// ---- Tensor finalizer ----

static void tensor_finalizer(SEXP ptr) {
    auto* t = (at::Tensor*)R_ExternalPtrAddr(ptr);
    if (t) {
        delete t;
        R_ClearExternalPtr(ptr);
    }
}

// ---- Helpers ----

SEXP make_tensor_sexp(at::Tensor* t) {
    SEXP ptr = PROTECT(R_MakeExternalPtr(t, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, tensor_finalizer, TRUE);

    // Set class to "torch_tensor"
    SEXP cls = PROTECT(Rf_allocVector(STRSXP, 1));
    SET_STRING_ELT(cls, 0, Rf_mkChar("torch_tensor"));
    Rf_setAttrib(ptr, R_ClassSymbol, cls);

    UNPROTECT(2);
    return ptr;
}

at::Tensor* get_tensor_ptr(SEXP x) {
    auto* t = (at::Tensor*)R_ExternalPtrAddr(x);
    if (!t) {
        Rf_error("Column pointer is NULL (tensor has been freed)");
    }
    return t;
}

at::Scalar sexp_to_scalar(SEXP x) {
    if (Rf_isInteger(x)) {
        return at::Scalar(Rf_asInteger(x));
    }
    return at::Scalar(Rf_asReal(x));
}

c10::optional<at::ScalarType> sexp_to_dtype(SEXP dtype) {
    if (Rf_isNull(dtype)) {
        return c10::nullopt;
    }
    int code = Rf_asInteger(dtype);
    return static_cast<at::ScalarType>(code);
}

std::vector<int64_t> sexp_to_int_vec(SEXP x) {
    R_xlen_t n = Rf_xlength(x);
    std::vector<int64_t> out(n);
    if (Rf_isInteger(x)) {
        int* px = INTEGER(x);
        for (R_xlen_t i = 0; i < n; i++) out[i] = px[i];
    } else {
        double* px = REAL(x);
        for (R_xlen_t i = 0; i < n; i++) out[i] = static_cast<int64_t>(px[i]);
    }
    return out;
}

// ---- Creation functions ----

extern "C" SEXP C_torch_tensor(SEXP data, SEXP dtype_sexp, SEXP device_sexp) {
    try {
        auto dtype = sexp_to_dtype(dtype_sexp);

        at::Tensor t;
        R_xlen_t n = Rf_xlength(data);

        if (Rf_isReal(data)) {
            auto opts = at::TensorOptions().dtype(at::kDouble);
            t = torch::from_blob(REAL(data), {n}, opts).clone();
        } else if (Rf_isInteger(data)) {
            auto opts = at::TensorOptions().dtype(at::kInt);
            t = torch::from_blob(INTEGER(data), {n}, opts).clone();
        } else if (Rf_isLogical(data)) {
            // R logicals are int internally
            auto opts = at::TensorOptions().dtype(at::kInt);
            t = torch::from_blob(LOGICAL(data), {n}, opts).clone().to(at::kBool);
        } else {
            Rf_error("Unsupported data type for torch_tensor");
        }

        // Reshape if data has dim attribute
        SEXP dim = Rf_getAttrib(data, R_DimSymbol);
        if (!Rf_isNull(dim)) {
            auto dims = sexp_to_int_vec(dim);
            // R stores column-major; reverse dims for row-major interpretation
            // Actually, we need to match torch's convention:
            // R matrix dim = c(nrow, ncol), stored column-major
            // We reshape to (nrow, ncol) and the data is column-major
            // So we need to reshape then transpose for matrices
            t = t.reshape(at::IntArrayRef(dims.data(), dims.size()));
        }

        if (dtype.has_value()) {
            t = t.to(dtype.value());
        }

        return make_tensor_sexp(new at::Tensor(t));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_tensor_raw(SEXP data, SEXP dim_sexp, SEXP dtype_sexp) {
    try {
        auto dtype = sexp_to_dtype(dtype_sexp);
        R_xlen_t n = Rf_xlength(data);

        at::Tensor t;
        if (Rf_isReal(data)) {
            t = torch::from_blob(REAL(data), {n}, at::kDouble).clone();
        } else if (Rf_isInteger(data)) {
            t = torch::from_blob(INTEGER(data), {n}, at::kInt).clone();
        } else if (Rf_isLogical(data)) {
            t = torch::from_blob(LOGICAL(data), {n}, at::kInt).clone().to(at::kBool);
        } else {
            Rf_error("Unsupported data type");
        }

        if (!Rf_isNull(dim_sexp)) {
            auto dims = sexp_to_int_vec(dim_sexp);
            t = t.reshape(at::IntArrayRef(dims.data(), dims.size()));
        }

        if (dtype.has_value()) {
            t = t.to(dtype.value());
        }

        return make_tensor_sexp(new at::Tensor(t));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_zeros(SEXP size_sexp, SEXP dtype_sexp) {
    try {
        auto size = sexp_to_int_vec(size_sexp);
        auto opts = at::TensorOptions();
        auto dtype = sexp_to_dtype(dtype_sexp);
        if (dtype.has_value()) opts = opts.dtype(dtype.value());
        auto* t = new at::Tensor(torch::zeros(at::IntArrayRef(size.data(), size.size()), opts));
        return make_tensor_sexp(t);
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_ones(SEXP size_sexp, SEXP dtype_sexp) {
    try {
        auto size = sexp_to_int_vec(size_sexp);
        auto opts = at::TensorOptions();
        auto dtype = sexp_to_dtype(dtype_sexp);
        if (dtype.has_value()) opts = opts.dtype(dtype.value());
        auto* t = new at::Tensor(torch::ones(at::IntArrayRef(size.data(), size.size()), opts));
        return make_tensor_sexp(t);
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_randn(SEXP size_sexp, SEXP dtype_sexp) {
    try {
        auto size = sexp_to_int_vec(size_sexp);
        auto opts = at::TensorOptions();
        auto dtype = sexp_to_dtype(dtype_sexp);
        if (dtype.has_value()) opts = opts.dtype(dtype.value());
        auto* t = new at::Tensor(torch::randn(at::IntArrayRef(size.data(), size.size()), opts));
        return make_tensor_sexp(t);
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_empty_like(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(at::empty_like(*a)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_empty(SEXP size_sexp, SEXP dtype_sexp) {
    try {
        auto size = sexp_to_int_vec(size_sexp);
        auto opts = at::TensorOptions();
        auto dtype = sexp_to_dtype(dtype_sexp);
        if (dtype.has_value()) opts = opts.dtype(dtype.value());
        auto* t = new at::Tensor(torch::empty(at::IntArrayRef(size.data(), size.size()), opts));
        return make_tensor_sexp(t);
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

// ---- Create tensor from raw bytes ----

extern "C" SEXP C_torch_tensor_from_buffer(SEXP raw_sexp, SEXP shape_sexp,
                                            SEXP dtype_sexp) {
    try {
        R_xlen_t nbytes = Rf_xlength(raw_sexp);
        void* data = RAW(raw_sexp);
        auto shape = sexp_to_int_vec(shape_sexp);
        auto dtype = sexp_to_dtype(dtype_sexp);
        if (!dtype.has_value()) {
            Rf_error("dtype must be specified for tensor_from_buffer");
        }

        auto opts = at::TensorOptions().dtype(dtype.value());
        at::Tensor t = torch::from_blob(
            data,
            at::IntArrayRef(shape.data(), shape.size()),
            opts
        ).clone();  // clone to own the data

        return make_tensor_sexp(new at::Tensor(t));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

// ---- New creation functions ----

extern "C" SEXP C_torch_arange(SEXP start_sexp, SEXP end_sexp,
                                SEXP step_sexp, SEXP dtype_sexp) {
    try {
        double start = Rf_asReal(start_sexp);
        double end = Rf_asReal(end_sexp);
        double step = Rf_asReal(step_sexp);
        // Match torch R package behavior: inclusive end
        // ATen arange is exclusive, so compute how many elements we need
        // and set end_exclusive = start + n * step
        int64_t n = (int64_t)std::floor((end - start) / step) + 1;
        end = start + n * step;
        auto opts = at::TensorOptions();
        auto dtype = sexp_to_dtype(dtype_sexp);
        if (dtype.has_value()) opts = opts.dtype(dtype.value());
        return make_tensor_sexp(new at::Tensor(
            torch::arange(start, end, step, opts)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_full(SEXP size_sexp, SEXP fill_sexp, SEXP dtype_sexp) {
    try {
        auto size = sexp_to_int_vec(size_sexp);
        double fill = Rf_asReal(fill_sexp);
        auto opts = at::TensorOptions();
        auto dtype = sexp_to_dtype(dtype_sexp);
        if (dtype.has_value()) opts = opts.dtype(dtype.value());
        return make_tensor_sexp(new at::Tensor(
            torch::full(at::IntArrayRef(size.data(), size.size()), fill, opts)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_linspace(SEXP start_sexp, SEXP end_sexp,
                                  SEXP steps_sexp, SEXP dtype_sexp) {
    try {
        double start = Rf_asReal(start_sexp);
        double end = Rf_asReal(end_sexp);
        int64_t steps = static_cast<int64_t>(Rf_asInteger(steps_sexp));
        auto opts = at::TensorOptions();
        auto dtype = sexp_to_dtype(dtype_sexp);
        if (dtype.has_value()) opts = opts.dtype(dtype.value());
        return make_tensor_sexp(new at::Tensor(
            torch::linspace(start, end, steps, opts)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_ones_like(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(at::ones_like(*a)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_zeros_like(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(at::zeros_like(*a)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_randn_like(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(at::randn_like(*a)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

// ---- Tensor method backends ----

extern "C" SEXP C_torch_permute(SEXP self, SEXP dims_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        auto dims = sexp_to_int_vec(dims_sexp);
        // Convert 1-based to 0-based
        for (auto& d : dims) {
            if (d > 0) d = d - 1;
        }
        return make_tensor_sexp(new at::Tensor(
            a->permute(at::IntArrayRef(dims.data(), dims.size()))));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_expand(SEXP self, SEXP size_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        auto size = sexp_to_int_vec(size_sexp);
        return make_tensor_sexp(new at::Tensor(
            a->expand(at::IntArrayRef(size.data(), size.size()))));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_gather(SEXP self, SEXP dim_sexp, SEXP index) {
    try {
        auto* a = get_tensor_ptr(self);
        auto* idx = get_tensor_ptr(index);
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return make_tensor_sexp(new at::Tensor(a->gather(dim, *idx)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_masked_fill(SEXP self, SEXP mask, SEXP value_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        auto* m = get_tensor_ptr(mask);
        double value = Rf_asReal(value_sexp);
        return make_tensor_sexp(new at::Tensor(
            a->masked_fill(*m, at::Scalar(value))));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_masked_fill_(SEXP self, SEXP mask, SEXP value_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        auto* m = get_tensor_ptr(mask);
        double value = Rf_asReal(value_sexp);
        a->masked_fill_(*m, at::Scalar(value));
        return self;
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_copy_(SEXP self, SEXP src) {
    try {
        auto* a = get_tensor_ptr(self);
        auto* b = get_tensor_ptr(src);
        a->copy_(*b);
        return self;
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_normal_(SEXP self, SEXP mean_sexp, SEXP std_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        double mean = Rf_asReal(mean_sexp);
        double std = Rf_asReal(std_sexp);
        a->normal_(mean, std);
        return self;
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_uniform_(SEXP self, SEXP from_sexp, SEXP to_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        double from = Rf_asReal(from_sexp);
        double to = Rf_asReal(to_sexp);
        a->uniform_(from, to);
        return self;
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_zero_(SEXP self) {
    try {
        auto* a = get_tensor_ptr(self);
        a->zero_();
        return self;
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_fill_(SEXP self, SEXP value_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        double value = Rf_asReal(value_sexp);
        a->fill_(at::Scalar(value));
        return self;
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_repeat(SEXP self, SEXP sizes_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        auto sizes = sexp_to_int_vec(sizes_sexp);
        return make_tensor_sexp(new at::Tensor(
            a->repeat_symint(c10::fromIntArrayRefSlow(sizes))));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_repeat_interleave(SEXP self, SEXP repeats_sexp,
                                            SEXP dim_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        int64_t repeats = static_cast<int64_t>(Rf_asInteger(repeats_sexp));
        if (Rf_isNull(dim_sexp)) {
            return make_tensor_sexp(new at::Tensor(
                a->repeat_interleave(repeats)));
        }
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return make_tensor_sexp(new at::Tensor(
            a->repeat_interleave(repeats, dim)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_index_select(SEXP self, SEXP dim_sexp, SEXP index) {
    try {
        auto* a = get_tensor_ptr(self);
        auto* idx = get_tensor_ptr(index);
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return make_tensor_sexp(new at::Tensor(a->index_select(dim, *idx)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_narrow(SEXP self, SEXP dim_sexp, SEXP start_sexp,
                                SEXP length_sexp) {
    try {
        auto* a = get_tensor_ptr(self);
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        int64_t start = static_cast<int64_t>(Rf_asInteger(start_sexp));
        int64_t length = static_cast<int64_t>(Rf_asInteger(length_sexp));
        return make_tensor_sexp(new at::Tensor(a->narrow(dim, start, length)));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_torch_scatter_(SEXP self, SEXP dim_sexp, SEXP index, SEXP src) {
    try {
        auto* a = get_tensor_ptr(self);
        auto* idx = get_tensor_ptr(index);
        auto* s = get_tensor_ptr(src);
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        a->scatter_(dim, *idx, *s);
        return self;
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}
