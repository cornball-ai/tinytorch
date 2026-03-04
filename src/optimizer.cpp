#include "Rtorch.h"
#include <torch/optim.h>

static void optimizer_finalizer(SEXP ptr) {
    auto* opt = static_cast<torch::optim::Optimizer*>(R_ExternalPtrAddr(ptr));
    if (opt) {
        delete opt;
        R_ClearExternalPtr(ptr);
    }
}

static SEXP wrap_optimizer(torch::optim::Optimizer* opt) {
    SEXP ptr = PROTECT(R_MakeExternalPtr(opt, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, optimizer_finalizer, TRUE);
    UNPROTECT(1);
    return ptr;
}

static torch::optim::Optimizer* get_optimizer_ptr(SEXP x) {
    auto* p = static_cast<torch::optim::Optimizer*>(R_ExternalPtrAddr(x));
    if (!p) Rf_error("optimizer pointer is NULL");
    return p;
}

// [[Rcpp::export]]
SEXP C_optim_sgd(SEXP params_sexp, double lr, double momentum,
                  double dampening, double weight_decay, bool nesterov) {
    auto params = sexp_to_tensor_list(params_sexp);
    auto opts = torch::optim::SGDOptions(lr)
        .momentum(momentum).dampening(dampening)
        .weight_decay(weight_decay).nesterov(nesterov);
    return wrap_optimizer(new torch::optim::SGD(params, opts));
}

// [[Rcpp::export]]
SEXP C_optim_adam(SEXP params_sexp, double lr, double beta1, double beta2,
                   double eps, double weight_decay, bool amsgrad) {
    auto params = sexp_to_tensor_list(params_sexp);
    auto opts = torch::optim::AdamOptions(lr)
        .betas(std::make_tuple(beta1, beta2))
        .eps(eps).weight_decay(weight_decay).amsgrad(amsgrad);
    return wrap_optimizer(new torch::optim::Adam(params, opts));
}

// [[Rcpp::export]]
SEXP C_optim_adamw(SEXP params_sexp, double lr, double beta1, double beta2,
                    double eps, double weight_decay, bool amsgrad) {
    auto params = sexp_to_tensor_list(params_sexp);
    auto opts = torch::optim::AdamWOptions(lr)
        .betas(std::make_tuple(beta1, beta2))
        .eps(eps).weight_decay(weight_decay).amsgrad(amsgrad);
    return wrap_optimizer(new torch::optim::AdamW(params, opts));
}

// [[Rcpp::export]]
void C_optim_step(SEXP optim_sexp) {
    get_optimizer_ptr(optim_sexp)->step();
}

// [[Rcpp::export]]
void C_optim_zero_grad(SEXP optim_sexp, bool set_to_none) {
    get_optimizer_ptr(optim_sexp)->zero_grad(set_to_none);
}
