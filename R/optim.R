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
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' p <- nn_parameter(torch_randn(5))
#' opt <- optim_sgd(list(p), lr = 0.01)
#' loss <- (p ^ 2)$sum()
#' loss$backward()
#' opt$step()
#' opt$zero_grad()
#' }
#' }
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
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' p <- nn_parameter(torch_randn(5))
#' opt <- optim_adam(list(p), lr = 0.001)
#' loss <- (p ^ 2)$sum()
#' loss$backward()
#' opt$step()
#' opt$zero_grad()
#' }
#' }
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
#' @examples
#' \donttest{
#' if (torch_is_installed()) {
#' p <- nn_parameter(torch_randn(5))
#' opt <- optim_adamw(list(p), lr = 0.001)
#' loss <- (p ^ 2)$sum()
#' loss$backward()
#' opt$step()
#' opt$zero_grad()
#' }
#' }
optim_adamw <- function(params, lr = 0.001, betas = c(0.9, 0.999),
                         eps = 1e-8, weight_decay = 0.01, amsgrad = FALSE) {
  if (is.list(params)) {
    params <- Filter(function(p) inherits(p, "torch_tensor"), params)
  }
  ptr <- C_optim_adamw(params, lr, betas[1], betas[2], eps, weight_decay, amsgrad)
  structure(list(ptr = ptr, param_list = params), class = "torch_optimizer")
}

#' @export

#' @return A `torch_tensor`.
#' @export
`$.torch_optimizer` <- function(x, name) {
  # Pure-R optimizers (no ptr, have .step_fn)
  if (is.null(x[["ptr"]])) {
    if (name == "zero_grad") {
      return(function(set_to_none = TRUE) {
        for (p in x$param_list) {
          g <- p$grad
          if (!is.null(g)) g$zero_()
        }
        invisible(x)
      })
    }
    if (name == "step") {
      return(function(closure = NULL) {
        # In-place updates on leaf parameters require autograd to be off,
        # exactly like torch.optim does inside torch.no_grad().
        with_no_grad(x$.step_fn(x))
        invisible(x)
      })
    }
    return(x[[name]])
  }
  # C++-backed optimizers (SGD, Adam, AdamW)
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
