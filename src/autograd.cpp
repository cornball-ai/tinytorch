#include "tinytorch.h"
#include <torch/csrc/autograd/autograd.h>

// [[Rcpp::export]]
SEXP C_tensor_requires_grad_(SEXP self_sexp, bool requires_grad) {
    at::Tensor* t = get_tensor_ptr(self_sexp);
    t->requires_grad_(requires_grad);
    return self_sexp;
}

// Compute gradients of `outputs` w.r.t. `inputs` without accumulating into
// the `.grad` field of inputs. Mirrors torch.autograd.grad.
//
// grad_outputs: R list of tensors, or NULL to default to ones-like for scalar
// outputs. retain_graph / create_graph / allow_unused map 1:1 to PyTorch.
// [[Rcpp::export]]
SEXP C_autograd_grad(SEXP outputs_sexp, SEXP inputs_sexp, SEXP grad_outputs_sexp,
                     bool retain_graph, bool create_graph, bool allow_unused) {
    std::vector<at::Tensor> outputs = sexp_to_tensor_list(outputs_sexp);
    std::vector<at::Tensor> inputs  = sexp_to_tensor_list(inputs_sexp);

    std::vector<at::Tensor> grad_outputs;
    if (!Rf_isNull(grad_outputs_sexp)) {
        grad_outputs = sexp_to_tensor_list(grad_outputs_sexp);
    }

    auto grads = torch::autograd::grad(
        outputs,
        inputs,
        grad_outputs,
        retain_graph,
        create_graph,
        allow_unused);

    return tensor_list_to_sexp(grads);
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
