// GPU kernel launch for Rtorch
// Loads compiled PTX, extracts device pointers from Rtorch tensors,
// and launches kernels via CUDA driver API.

#include "Rtorch.h"

#ifdef RTORCH_CUDA
#include <cuda.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdio>

#define CUDA_CHECK(call) do { \
  CUresult err = call; \
  if (err != CUDA_SUCCESS) { \
    const char *name, *msg; \
    cuGetErrorName(err, &name); \
    cuGetErrorString(err, &msg); \
    Rf_error("CUDA error: %s: %s", name, msg); \
  } \
} while(0)

// Lazy CUDA driver init
static bool gpu_initialized = false;
static CUcontext gpu_context = nullptr;

static void ensure_gpu_initialized() {
    if (gpu_initialized) return;
    CUDA_CHECK(cuInit(0));
    CUdevice device;
    CUDA_CHECK(cuDeviceGet(&device, 0));
    CUDA_CHECK(cuCtxCreate(&gpu_context, 0, device));
    gpu_initialized = true;
}

// Kernel cache: kernel_name -> {module, function}
struct CachedKernel {
    CUmodule module;
    CUfunction function;
};
static std::unordered_map<std::string, CachedKernel> kernel_cache;

static CUfunction get_cached_kernel(const std::string& ptx,
                                     const std::string& kernel_name) {
    auto it = kernel_cache.find(kernel_name);
    if (it != kernel_cache.end()) return it->second.function;

    CUmodule mod;
    CUDA_CHECK(cuModuleLoadData(&mod, ptx.c_str()));
    CUfunction func;
    CUDA_CHECK(cuModuleGetFunction(&func, mod, kernel_name.c_str()));
    kernel_cache[kernel_name] = {mod, func};
    return func;
}

// Extract CUDA device pointer from Rtorch tensor (at::Tensor*)
static CUdeviceptr get_device_ptr(SEXP tensor_sexp) {
    at::Tensor* t = get_tensor_ptr(tensor_sexp);
    if (!t->is_cuda()) {
        Rf_error("Tensor must be on CUDA device");
    }
    return reinterpret_cast<CUdeviceptr>(t->data_ptr());
}

// ================================================================
// Elementwise kernel launch
// ================================================================

extern "C" SEXP C_gpu_launch(
    SEXP ptx_sexp,          // character: PTX text
    SEXP kernel_name_sexp,  // character: kernel entry point
    SEXP inputs_sexp,       // list of torch_tensor
    SEXP output_sexp,       // torch_tensor (pre-allocated)
    SEXP grid_sexp,         // integer(3): grid dimensions
    SEXP block_sexp,        // integer(3): block dimensions
    SEXP shared_mem_sexp)   // integer: shared memory bytes
{
    try {
        ensure_gpu_initialized();

        std::string ptx(CHAR(STRING_ELT(ptx_sexp, 0)));
        std::string kernel_name(CHAR(STRING_ELT(kernel_name_sexp, 0)));

        CUfunction func = get_cached_kernel(ptx, kernel_name);

        // Extract input device pointers
        int n_inputs = Rf_length(inputs_sexp);
        std::vector<CUdeviceptr> input_ptrs(n_inputs);
        for (int i = 0; i < n_inputs; i++) {
            input_ptrs[i] = get_device_ptr(VECTOR_ELT(inputs_sexp, i));
        }

        // Output device pointer
        CUdeviceptr out_ptr = get_device_ptr(output_sexp);

        // n_elements from output tensor
        at::Tensor* out_t = get_tensor_ptr(output_sexp);
        int32_t n_elements = static_cast<int32_t>(out_t->numel());

        // Build args: [input_ptrs..., out_ptr, n_elements, null, null]
        // Triton adds 2 metadata pointers at the end
        CUdeviceptr null_ptr = 0;
        std::vector<void*> args;
        for (int i = 0; i < n_inputs; i++) {
            args.push_back(&input_ptrs[i]);
        }
        args.push_back(&out_ptr);
        args.push_back(&n_elements);
        args.push_back(&null_ptr);
        args.push_back(&null_ptr);

        int* grid = INTEGER(grid_sexp);
        int* block = INTEGER(block_sexp);
        int shared_mem = Rf_asInteger(shared_mem_sexp);

        CUDA_CHECK(cuLaunchKernel(
            func,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
            shared_mem, nullptr,
            args.data(), nullptr
        ));

        CUDA_CHECK(cuCtxSynchronize());

        return R_NilValue;
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

// ================================================================
// Reduction kernel launch
// ================================================================

extern "C" SEXP C_gpu_launch_reduction(
    SEXP ptx_sexp,
    SEXP kernel_name_sexp,
    SEXP input_sexp,        // single torch_tensor input
    SEXP output_sexp,       // torch_tensor output (reduced)
    SEXP n_elements_sexp,   // integer: total elements per reduction
    SEXP grid_sexp,
    SEXP block_sexp,
    SEXP shared_mem_sexp)
{
    try {
        ensure_gpu_initialized();

        std::string ptx(CHAR(STRING_ELT(ptx_sexp, 0)));
        std::string kernel_name(CHAR(STRING_ELT(kernel_name_sexp, 0)));

        CUfunction func = get_cached_kernel(ptx, kernel_name);

        CUdeviceptr in_ptr = get_device_ptr(input_sexp);
        CUdeviceptr out_ptr = get_device_ptr(output_sexp);
        int32_t n_elements = Rf_asInteger(n_elements_sexp);

        CUdeviceptr null_ptr = 0;
        void* args[] = {
            &in_ptr, &out_ptr, &n_elements,
            &null_ptr, &null_ptr
        };

        int* grid = INTEGER(grid_sexp);
        int* block = INTEGER(block_sexp);
        int shared_mem = Rf_asInteger(shared_mem_sexp);

        CUDA_CHECK(cuLaunchKernel(
            func,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
            shared_mem, nullptr,
            args, nullptr
        ));

        CUDA_CHECK(cuCtxSynchronize());

        return R_NilValue;
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

// ================================================================
// General-purpose kernel launch
// ================================================================

// Flexible launcher: takes list of tensors + list of scalar args.
// Args are packed in order: [tensor_ptrs..., scalar_args...]
// Scalars can be integer (i32) or double (f32).
extern "C" SEXP C_gpu_launch_generic(
    SEXP ptx_sexp,
    SEXP kernel_name_sexp,
    SEXP tensors_sexp,      // list of torch_tensor (device pointers)
    SEXP scalars_sexp,      // list of scalar values (integer or double)
    SEXP grid_sexp,
    SEXP block_sexp,
    SEXP shared_mem_sexp)
{
    try {
        ensure_gpu_initialized();

        std::string ptx(CHAR(STRING_ELT(ptx_sexp, 0)));
        std::string kernel_name(CHAR(STRING_ELT(kernel_name_sexp, 0)));

        CUfunction func = get_cached_kernel(ptx, kernel_name);

        // Extract tensor device pointers
        int n_tensors = Rf_length(tensors_sexp);
        std::vector<CUdeviceptr> tensor_ptrs(n_tensors);
        for (int i = 0; i < n_tensors; i++) {
            tensor_ptrs[i] = get_device_ptr(VECTOR_ELT(tensors_sexp, i));
        }

        // Extract scalar arguments
        int n_scalars = Rf_length(scalars_sexp);
        std::vector<int32_t> int_scalars;
        std::vector<float> float_scalars;
        // Track which type each scalar is
        struct ScalarArg { enum { INT, FLOAT } type; int idx; };
        std::vector<ScalarArg> scalar_info;

        for (int i = 0; i < n_scalars; i++) {
            SEXP s = VECTOR_ELT(scalars_sexp, i);
            if (Rf_isInteger(s)) {
                scalar_info.push_back({ScalarArg::INT,
                                       static_cast<int>(int_scalars.size())});
                int_scalars.push_back(INTEGER(s)[0]);
            } else {
                scalar_info.push_back({ScalarArg::FLOAT,
                                       static_cast<int>(float_scalars.size())});
                float_scalars.push_back(static_cast<float>(REAL(s)[0]));
            }
        }

        // Build args array: tensor pointers first, then scalars in order,
        // then 2 null metadata pointers (Triton convention)
        CUdeviceptr null_ptr = 0;
        std::vector<void*> args;
        for (int i = 0; i < n_tensors; i++) {
            args.push_back(&tensor_ptrs[i]);
        }
        for (int i = 0; i < n_scalars; i++) {
            if (scalar_info[i].type == ScalarArg::INT) {
                args.push_back(&int_scalars[scalar_info[i].idx]);
            } else {
                args.push_back(&float_scalars[scalar_info[i].idx]);
            }
        }
        args.push_back(&null_ptr);
        args.push_back(&null_ptr);

        int* grid = INTEGER(grid_sexp);
        int* block = INTEGER(block_sexp);
        int shared_mem = Rf_asInteger(shared_mem_sexp);

        CUDA_CHECK(cuLaunchKernel(
            func,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
            shared_mem, nullptr,
            args.data(), nullptr
        ));

        CUDA_CHECK(cuCtxSynchronize());

        return R_NilValue;
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

// ================================================================
// Cache management
// ================================================================

extern "C" SEXP C_gpu_kernel_cache_clear() {
    try {
        int n = static_cast<int>(kernel_cache.size());
        for (auto& kv : kernel_cache) {
            cuModuleUnload(kv.second.module);
        }
        kernel_cache.clear();
        return Rf_ScalarInteger(n);
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

extern "C" SEXP C_gpu_kernel_cache_stats() {
    try {
        int n = static_cast<int>(kernel_cache.size());
        SEXP names = PROTECT(Rf_allocVector(STRSXP, n));
        int i = 0;
        for (auto& kv : kernel_cache) {
            SET_STRING_ELT(names, i++, Rf_mkChar(kv.first.c_str()));
        }

        SEXP result = PROTECT(Rf_allocVector(VECSXP, 2));
        SET_VECTOR_ELT(result, 0, Rf_ScalarInteger(n));
        SET_VECTOR_ELT(result, 1, names);

        SEXP rnames = PROTECT(Rf_allocVector(STRSXP, 2));
        SET_STRING_ELT(rnames, 0, Rf_mkChar("n_cached"));
        SET_STRING_ELT(rnames, 1, Rf_mkChar("kernel_names"));
        Rf_setAttrib(result, R_NamesSymbol, rnames);

        UNPROTECT(3);
        return result;
    } catch (const std::exception& e) {
        Rf_error("%s", e.what());
    }
    return R_NilValue;
}

#else
// Non-CUDA stubs
extern "C" SEXP C_gpu_launch(SEXP a, SEXP b, SEXP c, SEXP d, SEXP e, SEXP f, SEXP g) {
    Rf_error("GPU launch requires CUDA support. Rebuild with CUDA."); return R_NilValue;
}
extern "C" SEXP C_gpu_launch_reduction(SEXP a, SEXP b, SEXP c, SEXP d, SEXP e, SEXP f, SEXP g, SEXP h) {
    Rf_error("GPU launch requires CUDA support. Rebuild with CUDA."); return R_NilValue;
}
extern "C" SEXP C_gpu_launch_generic(SEXP a, SEXP b, SEXP c, SEXP d, SEXP e, SEXP f, SEXP g) {
    Rf_error("GPU launch requires CUDA support. Rebuild with CUDA."); return R_NilValue;
}
extern "C" SEXP C_gpu_kernel_cache_clear() {
    return Rf_ScalarInteger(0);
}
extern "C" SEXP C_gpu_kernel_cache_stats() {
    SEXP result = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(result, 0, Rf_ScalarInteger(0));
    SET_VECTOR_ELT(result, 1, Rf_allocVector(STRSXP, 0));
    SEXP rnames = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(rnames, 0, Rf_mkChar("n_cached"));
    SET_STRING_ELT(rnames, 1, Rf_mkChar("kernel_names"));
    Rf_setAttrib(result, R_NamesSymbol, rnames);
    UNPROTECT(2);
    return result;
}
#endif
