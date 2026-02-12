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

// ---- Device helper ----

at::Device sexp_to_device(SEXP device_sexp) {
    if (Rf_isNull(device_sexp)) return at::Device(at::kCPU);
    return at::Device(std::string(CHAR(STRING_ELT(device_sexp, 0))));
}

// ---- Creation functions ----

// [[Rcpp::export]]
SEXP C_torch_tensor(SEXP data, SEXP dtype_sexp, SEXP device_sexp) {
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
            int ndim = dims.size();
            // R stores column-major (Fortran order), torch uses row-major (C order).
            // To convert: reshape with reversed dims, then permute to restore order.
            // Example: R matrix (2,3) stored as [col1, col2, col3] in memory.
            // Reshape as (3,2) reads cols as rows, then transpose gives (2,3) correct.
            std::vector<int64_t> rev_dims(dims.rbegin(), dims.rend());
            t = t.reshape(at::IntArrayRef(rev_dims.data(), rev_dims.size()));
            if (ndim > 1) {
                std::vector<int64_t> perm(ndim);
                for (int i = 0; i < ndim; i++) perm[i] = ndim - 1 - i;
                t = t.permute(at::IntArrayRef(perm.data(), perm.size())).contiguous();
            }
        }

        if (dtype.has_value()) {
            t = t.to(dtype.value());
        }

        if (!Rf_isNull(device_sexp)) {
            t = t.to(sexp_to_device(device_sexp));
        }

        return make_tensor_sexp(new at::Tensor(t));
}

// [[Rcpp::export]]
SEXP C_torch_tensor_raw(SEXP data, SEXP dim_sexp, SEXP dtype_sexp) {
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
}

// [[Rcpp::export]]
SEXP C_torch_zeros(SEXP size_sexp, SEXP dtype_sexp, SEXP device_sexp) {
        auto size = sexp_to_int_vec(size_sexp);
        auto opts = at::TensorOptions();
        auto dtype = sexp_to_dtype(dtype_sexp);
        if (dtype.has_value()) opts = opts.dtype(dtype.value());
        if (!Rf_isNull(device_sexp)) opts = opts.device(sexp_to_device(device_sexp));
        auto* t = new at::Tensor(torch::zeros(at::IntArrayRef(size.data(), size.size()), opts));
        return make_tensor_sexp(t);
}

// [[Rcpp::export]]
SEXP C_torch_ones(SEXP size_sexp, SEXP dtype_sexp, SEXP device_sexp) {
        auto size = sexp_to_int_vec(size_sexp);
        auto opts = at::TensorOptions();
        auto dtype = sexp_to_dtype(dtype_sexp);
        if (dtype.has_value()) opts = opts.dtype(dtype.value());
        if (!Rf_isNull(device_sexp)) opts = opts.device(sexp_to_device(device_sexp));
        auto* t = new at::Tensor(torch::ones(at::IntArrayRef(size.data(), size.size()), opts));
        return make_tensor_sexp(t);
}

// [[Rcpp::export]]
SEXP C_torch_randn(SEXP size_sexp, SEXP dtype_sexp, SEXP device_sexp) {
        auto size = sexp_to_int_vec(size_sexp);
        auto opts = at::TensorOptions();
        auto dtype = sexp_to_dtype(dtype_sexp);
        if (dtype.has_value()) opts = opts.dtype(dtype.value());
        if (!Rf_isNull(device_sexp)) opts = opts.device(sexp_to_device(device_sexp));
        auto* t = new at::Tensor(torch::randn(at::IntArrayRef(size.data(), size.size()), opts));
        return make_tensor_sexp(t);
}

// [[Rcpp::export]]
SEXP C_torch_empty_like(SEXP self) {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(at::empty_like(*a)));
}

// [[Rcpp::export]]
SEXP C_torch_empty(SEXP size_sexp, SEXP dtype_sexp, SEXP device_sexp) {
        auto size = sexp_to_int_vec(size_sexp);
        auto opts = at::TensorOptions();
        auto dtype = sexp_to_dtype(dtype_sexp);
        if (dtype.has_value()) opts = opts.dtype(dtype.value());
        if (!Rf_isNull(device_sexp)) opts = opts.device(sexp_to_device(device_sexp));
        auto* t = new at::Tensor(torch::empty(at::IntArrayRef(size.data(), size.size()), opts));
        return make_tensor_sexp(t);
}

// ---- Create tensor from raw bytes ----

// [[Rcpp::export]]
SEXP C_torch_tensor_from_buffer(SEXP raw_sexp, SEXP shape_sexp,
                                            SEXP dtype_sexp, SEXP device_sexp) {
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

        if (!Rf_isNull(device_sexp)) {
            t = t.to(sexp_to_device(device_sexp));
        }

        return make_tensor_sexp(new at::Tensor(t));
}

// ---- New creation functions ----

// [[Rcpp::export]]
SEXP C_torch_arange(SEXP start_sexp, SEXP end_sexp,
                                SEXP step_sexp, SEXP dtype_sexp,
                                SEXP device_sexp) {
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
        if (!Rf_isNull(device_sexp)) opts = opts.device(sexp_to_device(device_sexp));
        return make_tensor_sexp(new at::Tensor(
            torch::arange(start, end, step, opts)));
}

// [[Rcpp::export]]
SEXP C_torch_full(SEXP size_sexp, SEXP fill_sexp, SEXP dtype_sexp,
                              SEXP device_sexp) {
        auto size = sexp_to_int_vec(size_sexp);
        double fill = Rf_asReal(fill_sexp);
        auto opts = at::TensorOptions();
        auto dtype = sexp_to_dtype(dtype_sexp);
        if (dtype.has_value()) opts = opts.dtype(dtype.value());
        if (!Rf_isNull(device_sexp)) opts = opts.device(sexp_to_device(device_sexp));
        return make_tensor_sexp(new at::Tensor(
            torch::full(at::IntArrayRef(size.data(), size.size()), fill, opts)));
}

// [[Rcpp::export]]
SEXP C_torch_linspace(SEXP start_sexp, SEXP end_sexp,
                                  SEXP steps_sexp, SEXP dtype_sexp,
                                  SEXP device_sexp) {
        double start = Rf_asReal(start_sexp);
        double end = Rf_asReal(end_sexp);
        int64_t steps = static_cast<int64_t>(Rf_asInteger(steps_sexp));
        auto opts = at::TensorOptions();
        auto dtype = sexp_to_dtype(dtype_sexp);
        if (dtype.has_value()) opts = opts.dtype(dtype.value());
        if (!Rf_isNull(device_sexp)) opts = opts.device(sexp_to_device(device_sexp));
        return make_tensor_sexp(new at::Tensor(
            torch::linspace(start, end, steps, opts)));
}

// [[Rcpp::export]]
SEXP C_torch_ones_like(SEXP self, SEXP dtype_sexp) {
        auto* a = get_tensor_ptr(self);
        if (Rf_isNull(dtype_sexp)) {
            return make_tensor_sexp(new at::Tensor(at::ones_like(*a)));
        } else {
            auto dt = static_cast<c10::ScalarType>(Rf_asInteger(dtype_sexp));
            return make_tensor_sexp(new at::Tensor(at::ones_like(*a, at::TensorOptions().dtype(dt))));
        }
}

// [[Rcpp::export]]
SEXP C_torch_zeros_like(SEXP self, SEXP dtype_sexp) {
        auto* a = get_tensor_ptr(self);
        if (Rf_isNull(dtype_sexp)) {
            return make_tensor_sexp(new at::Tensor(at::zeros_like(*a)));
        } else {
            auto dt = static_cast<c10::ScalarType>(Rf_asInteger(dtype_sexp));
            return make_tensor_sexp(new at::Tensor(at::zeros_like(*a, at::TensorOptions().dtype(dt))));
        }
}

// [[Rcpp::export]]
SEXP C_torch_randn_like(SEXP self, SEXP dtype_sexp) {
        auto* a = get_tensor_ptr(self);
        if (Rf_isNull(dtype_sexp)) {
            return make_tensor_sexp(new at::Tensor(at::randn_like(*a)));
        } else {
            auto dt = static_cast<c10::ScalarType>(Rf_asInteger(dtype_sexp));
            return make_tensor_sexp(new at::Tensor(at::randn_like(*a, at::TensorOptions().dtype(dt))));
        }
}

// ---- Tensor method backends ----

// [[Rcpp::export]]
SEXP C_torch_permute(SEXP self, SEXP dims_sexp) {
        auto* a = get_tensor_ptr(self);
        auto dims = sexp_to_int_vec(dims_sexp);
        // Convert 1-based to 0-based
        for (auto& d : dims) {
            if (d > 0) d = d - 1;
        }
        return make_tensor_sexp(new at::Tensor(
            a->permute(at::IntArrayRef(dims.data(), dims.size()))));
}

// [[Rcpp::export]]
SEXP C_torch_expand(SEXP self, SEXP size_sexp) {
        auto* a = get_tensor_ptr(self);
        auto size = sexp_to_int_vec(size_sexp);
        return make_tensor_sexp(new at::Tensor(
            a->expand(at::IntArrayRef(size.data(), size.size()))));
}

// [[Rcpp::export]]
SEXP C_torch_gather(SEXP self, SEXP dim_sexp, SEXP index) {
        auto* a = get_tensor_ptr(self);
        auto* idx = get_tensor_ptr(index);
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        // Convert 1-indexed R indices to 0-indexed ATen indices
        auto idx0 = idx->sub(1);
        return make_tensor_sexp(new at::Tensor(a->gather(dim, idx0)));
}

// [[Rcpp::export]]
SEXP C_torch_masked_fill(SEXP self, SEXP mask, SEXP value_sexp) {
        auto* a = get_tensor_ptr(self);
        auto* m = get_tensor_ptr(mask);
        double value = Rf_asReal(value_sexp);
        return make_tensor_sexp(new at::Tensor(
            a->masked_fill(*m, at::Scalar(value))));
}

// [[Rcpp::export]]
SEXP C_torch_masked_fill_(SEXP self, SEXP mask, SEXP value_sexp) {
        auto* a = get_tensor_ptr(self);
        auto* m = get_tensor_ptr(mask);
        double value = Rf_asReal(value_sexp);
        a->masked_fill_(*m, at::Scalar(value));
        return self;
}

// [[Rcpp::export]]
SEXP C_torch_copy_(SEXP self, SEXP src) {
        auto* a = get_tensor_ptr(self);
        auto* b = get_tensor_ptr(src);
        a->copy_(*b);
        return self;
}

// [[Rcpp::export]]
SEXP C_torch_normal_(SEXP self, SEXP mean_sexp, SEXP std_sexp) {
        auto* a = get_tensor_ptr(self);
        double mean = Rf_asReal(mean_sexp);
        double std = Rf_asReal(std_sexp);
        a->normal_(mean, std);
        return self;
}

// [[Rcpp::export]]
SEXP C_torch_uniform_(SEXP self, SEXP from_sexp, SEXP to_sexp) {
        auto* a = get_tensor_ptr(self);
        double from = Rf_asReal(from_sexp);
        double to = Rf_asReal(to_sexp);
        a->uniform_(from, to);
        return self;
}

// [[Rcpp::export]]
SEXP C_torch_zero_(SEXP self) {
        auto* a = get_tensor_ptr(self);
        a->zero_();
        return self;
}

// [[Rcpp::export]]
SEXP C_torch_fill_(SEXP self, SEXP value_sexp) {
        auto* a = get_tensor_ptr(self);
        double value = Rf_asReal(value_sexp);
        a->fill_(at::Scalar(value));
        return self;
}

// [[Rcpp::export]]
SEXP C_torch_repeat(SEXP self, SEXP sizes_sexp) {
        auto* a = get_tensor_ptr(self);
        auto sizes = sexp_to_int_vec(sizes_sexp);
        return make_tensor_sexp(new at::Tensor(
            a->repeat_symint(c10::fromIntArrayRefSlow(sizes))));
}

// [[Rcpp::export]]
SEXP C_torch_repeat_interleave(SEXP self, SEXP repeats_sexp,
                                            SEXP dim_sexp) {
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
}

// [[Rcpp::export]]
SEXP C_torch_index_select(SEXP self, SEXP dim_sexp, SEXP index) {
        auto* a = get_tensor_ptr(self);
        auto* idx = get_tensor_ptr(index);
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return make_tensor_sexp(new at::Tensor(a->index_select(dim, *idx)));
}

// [[Rcpp::export]]
SEXP C_torch_narrow(SEXP self, SEXP dim_sexp, SEXP start_sexp,
                                SEXP length_sexp) {
        auto* a = get_tensor_ptr(self);
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        int64_t start = static_cast<int64_t>(Rf_asInteger(start_sexp));
        int64_t length = static_cast<int64_t>(Rf_asInteger(length_sexp));
        return make_tensor_sexp(new at::Tensor(a->narrow(dim, start, length)));
}

// [[Rcpp::export]]
SEXP C_torch_scatter_(SEXP self, SEXP dim_sexp, SEXP index, SEXP src) {
        auto* a = get_tensor_ptr(self);
        auto* idx = get_tensor_ptr(index);
        auto* s = get_tensor_ptr(src);
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        a->scatter_(dim, *idx, *s);
        return self;
}

// ---- Device transfer functions ----

// [[Rcpp::export]]
SEXP C_tensor_to_device(SEXP self, SEXP device_sexp) {
        auto* a = get_tensor_ptr(self);
        return make_tensor_sexp(new at::Tensor(a->to(sexp_to_device(device_sexp))));
}

// [[Rcpp::export]]
SEXP C_tensor_to_dtype_device(SEXP self, SEXP dtype_sexp,
                                          SEXP device_sexp) {
        auto* a = get_tensor_ptr(self);
        auto dtype = sexp_to_dtype(dtype_sexp);
        auto device = sexp_to_device(device_sexp);
        if (dtype.has_value()) {
            return make_tensor_sexp(new at::Tensor(
                a->to(device, dtype.value())));
        } else {
            return make_tensor_sexp(new at::Tensor(a->to(device)));
        }
}

// [[Rcpp::export]]
SEXP C_cuda_is_available() {
    return Rf_ScalarLogical(torch::cuda::is_available());
}

// [[Rcpp::export]]
SEXP C_cuda_device_count() {
    return Rf_ScalarInteger(static_cast<int>(torch::cuda::device_count()));
}

// ---- CUDA memory management ----

#ifdef RTORCH_CUDA
#ifndef RTORCH_CUDA_NO_SDK
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime_api.h>
#endif
#endif

// [[Rcpp::export]]
SEXP C_cuda_empty_cache() {
    if (!torch::cuda::is_available()) return R_NilValue;
#if defined(RTORCH_CUDA) && !defined(RTORCH_CUDA_NO_SDK)
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif
    return R_NilValue;
}

// Return c(free, total) in bytes for the current CUDA device
// [[Rcpp::export]]
SEXP C_cuda_mem_info() {
    if (!torch::cuda::is_available()) {
        return Rf_allocVector(REALSXP, 0);
    }
#if defined(RTORCH_CUDA) && !defined(RTORCH_CUDA_NO_SDK)
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) Rf_error("cudaMemGetInfo failed: %s", cudaGetErrorString(err));

    SEXP result = PROTECT(Rf_allocVector(REALSXP, 2));
    REAL(result)[0] = static_cast<double>(free_bytes);
    REAL(result)[1] = static_cast<double>(total_bytes);
    UNPROTECT(1);
    return result;
#else
    Rf_error("cuda_mem_info requires CUDA SDK headers at build time");
    return R_NilValue;
#endif
}

// Return c(allocated, reserved) in bytes from libtorch's caching allocator
// [[Rcpp::export]]
SEXP C_cuda_memory_stats() {
    if (!torch::cuda::is_available()) {
        return Rf_allocVector(REALSXP, 0);
    }
#if defined(RTORCH_CUDA) && !defined(RTORCH_CUDA_NO_SDK)
    try {
        auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
        SEXP result = PROTECT(Rf_allocVector(REALSXP, 4));
        REAL(result)[0] = static_cast<double>(stats.allocated_bytes[0].current);
        REAL(result)[1] = static_cast<double>(stats.allocated_bytes[0].peak);
        REAL(result)[2] = static_cast<double>(stats.reserved_bytes[0].current);
        REAL(result)[3] = static_cast<double>(stats.reserved_bytes[0].peak);
        UNPROTECT(1);
        return result;
    } catch (const std::exception& e) {
        // Allocator not yet initialized — return zeros
        SEXP result = PROTECT(Rf_allocVector(REALSXP, 4));
        REAL(result)[0] = REAL(result)[1] = REAL(result)[2] = REAL(result)[3] = 0.0;
        UNPROTECT(1);
        return result;
    }
#else
    Rf_error("cuda_memory_stats requires CUDA SDK headers at build time");
    return R_NilValue;
#endif
}
