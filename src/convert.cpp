#include "Rtorch.h"

extern "C" SEXP C_as_array(SEXP self) {
    try {
        auto* t = get_tensor_ptr(self);

        // Move to CPU
        at::Tensor cpu_t = t->to(at::kCPU);

        // Convert to double for R
        if (cpu_t.scalar_type() != at::kDouble) {
            cpu_t = cpu_t.to(at::kDouble);
        }

        int64_t numel = cpu_t.numel();
        int ndim = cpu_t.dim();

        // For multi-dimensional tensors, permute to column-major (R/Fortran order)
        // before copying data. Reverse dimension order so contiguous memory
        // matches R's column-major layout.
        if (ndim > 1) {
            std::vector<int64_t> perm(ndim);
            for (int i = 0; i < ndim; i++) perm[i] = ndim - 1 - i;
            cpu_t = cpu_t.permute(at::IntArrayRef(perm.data(), perm.size())).contiguous();
        } else {
            cpu_t = cpu_t.contiguous();
        }

        // Create R vector
        SEXP result = PROTECT(Rf_allocVector(REALSXP, numel));
        double* out = REAL(result);
        double* src = cpu_t.data_ptr<double>();
        std::memcpy(out, src, numel * sizeof(double));

        // Set dim attribute if multi-dimensional (use original sizes, not permuted)
        if (ndim > 1) {
            auto sizes = t->sizes();
            SEXP dim = PROTECT(Rf_allocVector(INTSXP, ndim));
            int* pdim = INTEGER(dim);
            for (int i = 0; i < ndim; i++) {
                pdim[i] = static_cast<int>(sizes[i]);
            }
            Rf_setAttrib(result, R_DimSymbol, dim);
            UNPROTECT(1);
        }

        UNPROTECT(1);
        return result;
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

// ---- Properties ----

extern "C" SEXP C_tensor_shape(SEXP self) {
    try {
        auto* t = get_tensor_ptr(self);
        auto sizes = t->sizes();
        int ndim = sizes.size();
        SEXP result = PROTECT(Rf_allocVector(INTSXP, ndim));
        int* out = INTEGER(result);
        for (int i = 0; i < ndim; i++) {
            out[i] = static_cast<int>(sizes[i]);
        }
        UNPROTECT(1);
        return result;
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_tensor_dtype(SEXP self) {
    try {
        auto* t = get_tensor_ptr(self);
        int code = static_cast<int>(t->scalar_type());
        return Rf_ScalarInteger(code);
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_tensor_device(SEXP self) {
    try {
        auto* t = get_tensor_ptr(self);
        std::string dev = t->device().str();
        return Rf_mkString(dev.c_str());
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_tensor_ndim(SEXP self) {
    try {
        auto* t = get_tensor_ptr(self);
        return Rf_ScalarInteger(t->dim());
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_tensor_numel(SEXP self) {
    try {
        auto* t = get_tensor_ptr(self);
        return Rf_ScalarInteger(static_cast<int>(t->numel()));
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_tensor_requires_grad(SEXP self) {
    try {
        auto* t = get_tensor_ptr(self);
        return Rf_ScalarLogical(t->requires_grad());
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_tensor_print(SEXP self) {
    try {
        auto* t = get_tensor_ptr(self);
        std::ostringstream oss;
        oss << *t;
        Rprintf("%s\n", oss.str().c_str());
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}
