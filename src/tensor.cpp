#include "tinytorch.h"

// ---- Backend availability ping ----

// [[Rcpp::export]]
SEXP C_rtorch_ping() {
    return Rf_ScalarInteger(1);
}

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

// ---- Optional tensor helper ----

c10::optional<at::Tensor> sexp_to_optional_tensor(SEXP x) {
    if (Rf_isNull(x)) return c10::nullopt;
    return *get_tensor_ptr(x);
}

// ---- Tensor list helper ----

std::vector<at::Tensor> sexp_to_tensor_list(SEXP x) {
    R_xlen_t n = Rf_xlength(x);
    std::vector<at::Tensor> out;
    out.reserve(n);
    for (R_xlen_t i = 0; i < n; i++)
        out.push_back(*get_tensor_ptr(VECTOR_ELT(x, i)));
    return out;
}

// ---- Optional primitive helpers ----

c10::optional<int64_t> sexp_to_optional_int(SEXP x) {
    if (Rf_isNull(x)) return c10::nullopt;
    return static_cast<int64_t>(Rf_asInteger(x));
}

c10::optional<double> sexp_to_optional_double(SEXP x) {
    if (Rf_isNull(x)) return c10::nullopt;
    return Rf_asReal(x);
}

c10::optional<bool> sexp_to_optional_bool(SEXP x) {
    if (Rf_isNull(x)) return c10::nullopt;
    return static_cast<bool>(Rf_asLogical(x));
}

// ---- Optional string helper ----

c10::optional<std::string> sexp_to_optional_string(SEXP x) {
    if (Rf_isNull(x)) return c10::nullopt;
    return std::string(CHAR(STRING_ELT(x, 0)));
}

// ---- Optional scalar helper ----

c10::optional<at::Scalar> sexp_to_optional_scalar(SEXP x) {
    if (Rf_isNull(x)) return c10::nullopt;
    return sexp_to_scalar(x);
}

// ---- Optional device helper ----

c10::optional<at::Device> sexp_to_optional_device(SEXP x) {
    if (Rf_isNull(x)) return c10::nullopt;
    return at::Device(std::string(CHAR(STRING_ELT(x, 0))));
}

// ---- Non-optional device helper ----

at::Device sexp_to_required_device(SEXP x) {
    return at::Device(std::string(CHAR(STRING_ELT(x, 0))));
}

// ---- Optional generator helper ----

c10::optional<at::Generator> sexp_to_optional_generator(SEXP x) {
    // Always pass nullopt — libtorch uses default RNG
    // R-side generator objects not yet supported
    return c10::nullopt;
}

// ---- Memory format helper ----

c10::optional<at::MemoryFormat> sexp_to_optional_memory_format(SEXP x) {
    if (Rf_isNull(x)) return c10::nullopt;
    std::string s = CHAR(STRING_ELT(x, 0));
    if (s == "contiguous") return at::MemoryFormat::Contiguous;
    if (s == "channels_last") return at::MemoryFormat::ChannelsLast;
    if (s == "channels_last_3d") return at::MemoryFormat::ChannelsLast3d;
    if (s == "preserve") return at::MemoryFormat::Preserve;
    Rf_error("Unknown memory format: %s", s.c_str());
    return c10::nullopt;
}

// ---- Double vector helper ----

std::vector<double> sexp_to_double_vec(SEXP x) {
    R_xlen_t n = Rf_xlength(x);
    std::vector<double> out(n);
    double* p = REAL(x);
    for (R_xlen_t i = 0; i < n; i++) out[i] = p[i];
    return out;
}

c10::optional<std::vector<double>> sexp_to_optional_double_vec(SEXP x) {
    if (Rf_isNull(x)) return c10::nullopt;
    return sexp_to_double_vec(x);
}

// ---- Scalar list helper ----

std::vector<at::Scalar> sexp_to_scalar_list(SEXP x) {
    R_xlen_t n = Rf_xlength(x);
    std::vector<at::Scalar> out;
    out.reserve(n);
    if (Rf_isReal(x)) {
        double* p = REAL(x);
        for (R_xlen_t i = 0; i < n; i++) out.emplace_back(p[i]);
    } else {
        int* p = INTEGER(x);
        for (R_xlen_t i = 0; i < n; i++) out.emplace_back(static_cast<int64_t>(p[i]));
    }
    return out;
}

// ---- Optional tensor list helper ----

c10::List<c10::optional<at::Tensor>> sexp_to_optional_tensor_list(SEXP x) {
    R_xlen_t n = Rf_xlength(x);
    c10::List<c10::optional<at::Tensor>> out;
    out.reserve(n);
    for (R_xlen_t i = 0; i < n; i++) {
        SEXP elt = VECTOR_ELT(x, i);
        if (Rf_isNull(elt)) {
            out.push_back(c10::nullopt);
        } else {
            out.push_back(*get_tensor_ptr(elt));
        }
    }
    return out;
}

// ---- Dimname helpers ----

at::Dimname sexp_to_dimname(SEXP x) {
    std::string s = CHAR(STRING_ELT(x, 0));
    return at::Dimname::fromSymbol(c10::Symbol::fromQualString(s));
}

std::vector<at::Dimname> sexp_to_dimname_vec(SEXP x) {
    R_xlen_t n = Rf_xlength(x);
    std::vector<at::Dimname> out;
    out.reserve(n);
    for (R_xlen_t i = 0; i < n; i++) {
        std::string s = CHAR(STRING_ELT(x, i));
        out.push_back(at::Dimname::fromSymbol(c10::Symbol::fromQualString(s)));
    }
    return out;
}

c10::optional<std::vector<at::Dimname>> sexp_to_optional_dimname_vec(SEXP x) {
    if (Rf_isNull(x)) return c10::nullopt;
    return sexp_to_dimname_vec(x);
}

// ---- Tensor list to SEXP helper (for Tensor[] returns) ----

SEXP tensor_list_to_sexp(const std::vector<at::Tensor>& tensors) {
    R_xlen_t n = tensors.size();
    SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
    for (R_xlen_t i = 0; i < n; i++) {
        SET_VECTOR_ELT(out, i, make_tensor_sexp(new at::Tensor(tensors[i])));
    }
    UNPROTECT(1);
    return out;
}

// ---- Rcpp::as / Rcpp::wrap for at::Tensor ----

namespace Rcpp {
    template<> at::Tensor as(SEXP x) {
        return *get_tensor_ptr(x);
    }
    template<> SEXP wrap(const at::Tensor& t) {
        return make_tensor_sexp(new at::Tensor(t));
    }
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
at::Tensor C_torch_empty_like(at::Tensor self) { return at::empty_like(self); }

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
at::Tensor C_torch_ones_like(at::Tensor self, SEXP dtype_sexp) {
        if (Rf_isNull(dtype_sexp)) {
            return at::ones_like(self);
        } else {
            auto dt = static_cast<c10::ScalarType>(Rf_asInteger(dtype_sexp));
            return at::ones_like(self, at::TensorOptions().dtype(dt));
        }
}

// [[Rcpp::export]]
at::Tensor C_torch_zeros_like(at::Tensor self, SEXP dtype_sexp) {
        if (Rf_isNull(dtype_sexp)) {
            return at::zeros_like(self);
        } else {
            auto dt = static_cast<c10::ScalarType>(Rf_asInteger(dtype_sexp));
            return at::zeros_like(self, at::TensorOptions().dtype(dt));
        }
}

// [[Rcpp::export]]
at::Tensor C_torch_randn_like(at::Tensor self, SEXP dtype_sexp) {
        if (Rf_isNull(dtype_sexp)) {
            return at::randn_like(self);
        } else {
            auto dt = static_cast<c10::ScalarType>(Rf_asInteger(dtype_sexp));
            return at::randn_like(self, at::TensorOptions().dtype(dt));
        }
}

// ---- Tensor method backends ----

// [[Rcpp::export]]
at::Tensor C_torch_permute(at::Tensor self, SEXP dims_sexp) {
        auto dims = sexp_to_int_vec(dims_sexp);
        // Convert 1-based to 0-based
        for (auto& d : dims) {
            if (d > 0) d = d - 1;
        }
        return self.permute(at::IntArrayRef(dims.data(), dims.size()));
}

// [[Rcpp::export]]
at::Tensor C_torch_expand(at::Tensor self, SEXP size_sexp) {
        auto size = sexp_to_int_vec(size_sexp);
        return self.expand(at::IntArrayRef(size.data(), size.size()));
}

// [[Rcpp::export]]
at::Tensor C_torch_gather(at::Tensor self, SEXP dim_sexp, at::Tensor index) {
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        // Convert 1-indexed R indices to 0-indexed ATen indices
        return self.gather(dim, index.sub(1));
}

// [[Rcpp::export]]
at::Tensor C_torch_masked_fill(at::Tensor self, at::Tensor mask, SEXP value_sexp) {
        double value = Rf_asReal(value_sexp);
        return self.masked_fill(mask, at::Scalar(value));
}

// [[Rcpp::export]]
SEXP C_torch_masked_fill_(SEXP self, at::Tensor mask, SEXP value_sexp) {
        auto* a = get_tensor_ptr(self);
        double value = Rf_asReal(value_sexp);
        a->masked_fill_(mask, at::Scalar(value));
        return self;
}

// [[Rcpp::export]]
SEXP C_torch_copy_(SEXP self, at::Tensor src) {
        auto* a = get_tensor_ptr(self);
        a->copy_(src);
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
at::Tensor C_torch_repeat(at::Tensor self, SEXP sizes_sexp) {
        auto sizes = sexp_to_int_vec(sizes_sexp);
        return self.repeat_symint(c10::fromIntArrayRefSlow(sizes));
}

// [[Rcpp::export]]
at::Tensor C_torch_repeat_interleave(at::Tensor self, SEXP repeats_sexp,
                                            SEXP dim_sexp) {
        int64_t repeats = static_cast<int64_t>(Rf_asInteger(repeats_sexp));
        if (Rf_isNull(dim_sexp)) {
            return self.repeat_interleave(repeats);
        }
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return self.repeat_interleave(repeats, dim);
}

// [[Rcpp::export]]
at::Tensor C_torch_index_select(at::Tensor self, SEXP dim_sexp, at::Tensor index) {
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        return self.index_select(dim, index);
}

// [[Rcpp::export]]
at::Tensor C_torch_narrow(at::Tensor self, SEXP dim_sexp, SEXP start_sexp,
                                SEXP length_sexp) {
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        int64_t start = static_cast<int64_t>(Rf_asInteger(start_sexp));
        int64_t length = static_cast<int64_t>(Rf_asInteger(length_sexp));
        return self.narrow(dim, start, length);
}

// [[Rcpp::export]]
SEXP C_torch_scatter_(SEXP self, SEXP dim_sexp, at::Tensor index, at::Tensor src) {
        auto* a = get_tensor_ptr(self);
        int64_t dim = static_cast<int64_t>(Rf_asInteger(dim_sexp));
        if (dim > 0) dim = dim - 1;
        a->scatter_(dim, index, src);
        return self;
}

// ---- Device transfer functions ----

// [[Rcpp::export]]
at::Tensor C_tensor_to_device(at::Tensor self, SEXP device_sexp) {
        return self.to(sexp_to_device(device_sexp));
}

// [[Rcpp::export]]
at::Tensor C_tensor_to_dtype_device(at::Tensor self, SEXP dtype_sexp,
                                          SEXP device_sexp) {
        auto dtype = sexp_to_dtype(dtype_sexp);
        auto device = sexp_to_device(device_sexp);
        if (dtype.has_value()) {
            return self.to(device, dtype.value());
        } else {
            return self.to(device);
        }
}

// ---- Thread management ----

// [[Rcpp::export]]
SEXP C_torch_set_num_threads(SEXP n_sexp) {
    at::set_num_threads(Rf_asInteger(n_sexp));
    return R_NilValue;
}

// [[Rcpp::export]]
SEXP C_torch_get_num_threads() {
    return Rf_ScalarInteger(at::get_num_threads());
}

// [[Rcpp::export]]
SEXP C_torch_set_num_interop_threads(SEXP n_sexp) {
    at::set_num_interop_threads(Rf_asInteger(n_sexp));
    return R_NilValue;
}

// [[Rcpp::export]]
SEXP C_torch_get_num_interop_threads() {
    return Rf_ScalarInteger(at::get_num_interop_threads());
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

#ifdef TINYTORCH_CUDA
#ifndef TINYTORCH_CUDA_NO_SDK
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime_api.h>
#endif
#endif

// [[Rcpp::export]]
SEXP C_cuda_empty_cache() {
    if (!torch::cuda::is_available()) return R_NilValue;
#if defined(TINYTORCH_CUDA) && !defined(TINYTORCH_CUDA_NO_SDK)
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif
    return R_NilValue;
}

// Block until all queued CUDA work on all devices completes.
// No-op when CUDA is unavailable.
// [[Rcpp::export]]
SEXP C_cuda_synchronize() {
    if (!torch::cuda::is_available()) return R_NilValue;
#if defined(TINYTORCH_CUDA) && !defined(TINYTORCH_CUDA_NO_SDK)
    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess) return R_NilValue;
    int saved = 0;
    cudaGetDevice(&saved);
    for (int i = 0; i < n_devices; ++i) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }
    cudaSetDevice(saved);
#endif
    return R_NilValue;
}

// Return c(free, total) in bytes for the current CUDA device
// [[Rcpp::export]]
SEXP C_cuda_mem_info() {
    if (!torch::cuda::is_available()) {
        return Rf_allocVector(REALSXP, 0);
    }
#if defined(TINYTORCH_CUDA) && !defined(TINYTORCH_CUDA_NO_SDK)
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
#if defined(TINYTORCH_CUDA) && !defined(TINYTORCH_CUDA_NO_SDK)
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
