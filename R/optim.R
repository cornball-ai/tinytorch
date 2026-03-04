# ---- Optimizers ----

#' SGD optimizer
#' @param params List of tensors to optimize.
#' @param lr Learning rate.
#' @param momentum Momentum factor. Default 0.
#' @param dampening Dampening for momentum. Default 0.
#' @param weight_decay Weight decay (L2 penalty). Default 0.
#' @param nesterov Whether to use Nesterov momentum. Default FALSE.
#' @return A torch_optimizer object.
#' @export
optim_sgd <- function(params, lr, momentum = 0, dampening = 0,
                       weight_decay = 0, nesterov = FALSE) {
  if (is.list(params)) {
    params <- Filter(function(p) inherits(p, "torch_tensor"), params)
  }
  ptr <- C_optim_sgd(params, lr, momentum, dampening, weight_decay, nesterov)
  structure(list(ptr = ptr, param_list = params), class = "torch_optimizer")
}

#' Adam optimizer
#' @param params List of tensors to optimize.
#' @param lr Learning rate. Default 0.001.
#' @param betas Coefficients for computing running averages. Default c(0.9, 0.999).
#' @param eps Term for numerical stability. Default 1e-8.
#' @param weight_decay Weight decay (L2 penalty). Default 0.
#' @param amsgrad Whether to use AMSGrad variant. Default FALSE.
#' @return A torch_optimizer object.
#' @export
optim_adam <- function(params, lr = 0.001, betas = c(0.9, 0.999),
                        eps = 1e-8, weight_decay = 0, amsgrad = FALSE) {
  if (is.list(params)) {
    params <- Filter(function(p) inherits(p, "torch_tensor"), params)
  }
  ptr <- C_optim_adam(params, lr, betas[1], betas[2], eps, weight_decay, amsgrad)
  structure(list(ptr = ptr, param_list = params), class = "torch_optimizer")
}

#' AdamW optimizer
#' @param params List of tensors to optimize.
#' @param lr Learning rate. Default 0.001.
#' @param betas Coefficients for computing running averages. Default c(0.9, 0.999).
#' @param eps Term for numerical stability. Default 1e-8.
#' @param weight_decay Weight decay. Default 0.01.
#' @param amsgrad Whether to use AMSGrad variant. Default FALSE.
#' @return A torch_optimizer object.
#' @export
optim_adamw <- function(params, lr = 0.001, betas = c(0.9, 0.999),
                         eps = 1e-8, weight_decay = 0.01, amsgrad = FALSE) {
  if (is.list(params)) {
    params <- Filter(function(p) inherits(p, "torch_tensor"), params)
  }
  ptr <- C_optim_adamw(params, lr, betas[1], betas[2], eps, weight_decay, amsgrad)
  structure(list(ptr = ptr, param_list = params), class = "torch_optimizer")
}

#' @export
`$.torch_optimizer` <- function(x, name) {
  if (name == "zero_grad") {
    return(function(set_to_none = TRUE) {
      C_optim_zero_grad(x$ptr, set_to_none)
      invisible(x)
    })
  }
  if (name == "step") {
    return(function() {
      C_optim_step(x$ptr)
      invisible(x)
    })
  }
  x[[name]]
}
