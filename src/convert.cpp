#include "Rtorch.h"

// [[Rcpp::export]]
SEXP C_as_array(at::Tensor self) {
        // Move to CPU
        at::Tensor cpu_t = self.to(at::kCPU);

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
            auto sizes = self.sizes();
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
}

// ---- Properties ----

// [[Rcpp::export]]
SEXP C_tensor_shape(at::Tensor self) {
        auto sizes = self.sizes();
        int ndim = sizes.size();
        SEXP result = PROTECT(Rf_allocVector(INTSXP, ndim));
        int* out = INTEGER(result);
        for (int i = 0; i < ndim; i++) {
            out[i] = static_cast<int>(sizes[i]);
        }
        UNPROTECT(1);
        return result;
}

// [[Rcpp::export]]
SEXP C_tensor_dtype(at::Tensor self) {
        int code = static_cast<int>(self.scalar_type());
        return Rf_ScalarInteger(code);
}

// [[Rcpp::export]]
SEXP C_tensor_device(at::Tensor self) {
        std::string dev = self.device().str();
        return Rf_mkString(dev.c_str());
}

// [[Rcpp::export]]
SEXP C_tensor_ndim(at::Tensor self) {
        return Rf_ScalarInteger(self.dim());
}

// [[Rcpp::export]]
SEXP C_tensor_numel(at::Tensor self) {
        return Rf_ScalarInteger(static_cast<int>(self.numel()));
}

// [[Rcpp::export]]
SEXP C_tensor_requires_grad(at::Tensor self) {
        return Rf_ScalarLogical(self.requires_grad());
}

// [[Rcpp::export]]
SEXP C_tensor_print(at::Tensor self) {
        std::ostringstream oss;
        oss << self;
        Rprintf("%s\n", oss.str().c_str());
}
