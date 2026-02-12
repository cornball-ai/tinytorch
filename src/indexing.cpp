#include "Rtorch.h"
#include <ATen/TensorIndexing.h>
#include <vector>

using namespace at::indexing;

// Check if an integer vector represents a contiguous range (e.g., 1:5)
static bool is_contiguous_range(int* data, R_xlen_t len) {
    if (len <= 1) return true;
    for (R_xlen_t i = 1; i < len; i++) {
        if (data[i] != data[0] + (int)i) return false;
    }
    return true;
}

// Convert a single R index argument to a TensorIndex
// R uses 1-based indexing; negative means "from end" (Python convention for torch)
static TensorIndex sexp_to_index(SEXP idx) {
    // NULL = select all (missing dimension)
    if (Rf_isNull(idx)) {
        return Slice();
    }

    // torch_tensor = advanced indexing (boolean mask or integer indices)
    if (Rf_inherits(idx, "torch_tensor")) {
        auto* t = get_tensor_ptr(idx);
        return *t;
    }

    // Logical vector = boolean mask
    if (Rf_isLogical(idx)) {
        R_xlen_t len = Rf_xlength(idx);
        int* lgl = LOGICAL(idx);
        // Convert to boolean tensor
        auto opts = at::TensorOptions().dtype(at::kBool);
        at::Tensor mask = at::empty({(int64_t)len}, opts);
        auto acc = mask.accessor<bool, 1>();
        for (R_xlen_t i = 0; i < len; i++) {
            acc[i] = (lgl[i] == TRUE);
        }
        return mask;
    }

    // Integer or numeric scalar/vector
    if (Rf_isInteger(idx) || Rf_isReal(idx)) {
        R_xlen_t len = Rf_xlength(idx);

        if (len == 1) {
            // Single index: scalar selection
            int64_t val = (int64_t)Rf_asInteger(idx);
            if (val > 0) {
                val -= 1;  // R 1-indexed to 0-indexed
            } else if (val < 0) {
                // Negative: count from end (-1 = last)
                // Already in Python convention, keep as-is
            } else {
                Rf_error("Index 0 is not valid (R uses 1-based indexing)");
            }
            return val;
        }

        // Vector of indices
        // First, get as integer array
        std::vector<int64_t> vals(len);
        if (Rf_isInteger(idx)) {
            int* idata = INTEGER(idx);
            // Check if contiguous range
            if (is_contiguous_range(idata, len)) {
                // Contiguous: use Slice (start 0-indexed, end exclusive 0-indexed)
                int64_t start = (int64_t)idata[0] - 1;  // 1-indexed to 0-indexed
                int64_t end = (int64_t)idata[len - 1];   // exclusive end = last + 1, already 0-indexed after subtracting then adding
                return Slice(start, end);
            }
            for (R_xlen_t i = 0; i < len; i++) {
                vals[i] = (int64_t)idata[i] - 1;  // 1-indexed to 0-indexed
            }
        } else {
            double* ddata = REAL(idx);
            // Check if contiguous
            bool contiguous = true;
            for (R_xlen_t i = 1; i < len && contiguous; i++) {
                if (ddata[i] != ddata[0] + (double)i) contiguous = false;
            }
            if (contiguous) {
                int64_t start = (int64_t)ddata[0] - 1;
                int64_t end = (int64_t)ddata[len - 1];
                return Slice(start, end);
            }
            for (R_xlen_t i = 0; i < len; i++) {
                vals[i] = (int64_t)ddata[i] - 1;
            }
        }

        // Non-contiguous: create index tensor
        auto opts = at::TensorOptions().dtype(at::kLong);
        at::Tensor indices = at::from_blob(vals.data(), {(int64_t)len}, opts).clone();
        return indices;
    }

    Rf_error("Unsupported index type");
    return Slice(); // unreachable
}

// C_torch_index: implements [.torch_tensor
// indices_list: R list where each element is an index spec (NULL, int, vec, logical, tensor)
// drop: whether to drop dimensions from scalar indexing
// [[Rcpp::export]]
SEXP C_torch_index(SEXP self_sexp, SEXP indices_list, SEXP drop_sexp) {
        auto* self = get_tensor_ptr(self_sexp);
        bool drop = Rf_asLogical(drop_sexp);
        R_xlen_t n_indices = Rf_xlength(indices_list);

        // Track which dimensions used scalar indexing (for drop=FALSE)
        std::vector<int64_t> scalar_dims;

        // Build vector of TensorIndex
        std::vector<TensorIndex> indices;
        indices.reserve(n_indices);

        for (R_xlen_t i = 0; i < n_indices; i++) {
            SEXP idx = VECTOR_ELT(indices_list, i);
            TensorIndex ti = sexp_to_index(idx);
            indices.push_back(ti);

            // Track scalar selections for drop behavior
            if (!Rf_isNull(idx) && !Rf_isLogical(idx) && !Rf_inherits(idx, "torch_tensor")) {
                if ((Rf_isInteger(idx) || Rf_isReal(idx)) && Rf_xlength(idx) == 1) {
                    scalar_dims.push_back((int64_t)i);
                }
            }
        }

        at::Tensor result = self->index(indices);

        // If drop=FALSE, unsqueeze the scalar-indexed dimensions back
        if (!drop && !scalar_dims.empty()) {
            for (size_t i = 0; i < scalar_dims.size(); i++) {
                result = result.unsqueeze(scalar_dims[i]);
            }
        }

        return make_tensor_sexp(new at::Tensor(result));
}

// C_torch_index_put: implements [<-.torch_tensor
// indices_list: same format as C_torch_index
// value: either a torch_tensor or a scalar
// [[Rcpp::export]]
SEXP C_torch_index_put(SEXP self_sexp, SEXP indices_list, SEXP value_sexp) {
        auto* self = get_tensor_ptr(self_sexp);
        R_xlen_t n_indices = Rf_xlength(indices_list);

        // Build vector of TensorIndex
        std::vector<TensorIndex> indices;
        indices.reserve(n_indices);

        for (R_xlen_t i = 0; i < n_indices; i++) {
            SEXP idx = VECTOR_ELT(indices_list, i);
            indices.push_back(sexp_to_index(idx));
        }

        if (Rf_inherits(value_sexp, "torch_tensor")) {
            auto* val = get_tensor_ptr(value_sexp);
            self->index_put_(indices, *val);
        } else if (Rf_isNumeric(value_sexp) || Rf_isInteger(value_sexp)) {
            at::Scalar s = sexp_to_scalar(value_sexp);
            self->index_put_(indices, s);
        } else if (Rf_isLogical(value_sexp)) {
            bool val = Rf_asLogical(value_sexp);
            self->index_put_(indices, at::Scalar(val));
        } else {
            Rf_error("Value must be a tensor, numeric, or logical");
        }

        return self_sexp;
}
