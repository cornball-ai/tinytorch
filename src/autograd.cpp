#include "Rtorch.h"

// [[Rcpp::export]]
SEXP C_tensor_requires_grad_(SEXP self_sexp, bool requires_grad) {
    at::Tensor* t = get_tensor_ptr(self_sexp);
    t->requires_grad_(requires_grad);
    return self_sexp;
}

// [[Rcpp::export]]
SEXP C_tensor_grad(SEXP self_sexp) {
    at::Tensor* t = get_tensor_ptr(self_sexp);
    at::Tensor g = t->grad();
    if (!g.defined()) return R_NilValue;
    return make_tensor_sexp(new at::Tensor(g));
}

// [[Rcpp::export]]
void C_tensor_backward(SEXP self_sexp, SEXP gradient_sexp,
                        bool retain_graph, bool create_graph) {
    at::Tensor* t = get_tensor_ptr(self_sexp);
    if (Rf_isNull(gradient_sexp)) {
        t->backward({}, retain_graph, create_graph);
    } else {
        t->backward(*get_tensor_ptr(gradient_sexp), retain_graph, create_graph);
    }
}

// [[Rcpp::export]]
void C_autograd_set_grad_mode(bool enabled) {
    at::GradMode::set_enabled(enabled);
}

// [[Rcpp::export]]
bool C_autograd_is_enabled() {
    return at::GradMode::is_enabled();
}

// [[Rcpp::export]]
bool C_tensor_is_leaf(SEXP self_sexp) {
    return get_tensor_ptr(self_sexp)->is_leaf();
}

// [[Rcpp::export]]
void C_tensor_retain_grad(SEXP self_sexp) {
    get_tensor_ptr(self_sexp)->retain_grad();
}
