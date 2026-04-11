# Pure-R optimizers
#
# These implement optimizer step logic in R using tensor ops.
# SGD/Adam/AdamW use C++ (optim.R); these cover the rest.
# Adapted from torch R package (MIT, Daniel Falbel).

# ---- Pure-R optimizer base ----

#' Create a pure-R optimizer
#'
#' @param params List of tensors to optimize (must have requires_grad).
#' @param defaults Named list of default hyperparameters.
#' @return An optimizer environment with step() and zero_grad() methods.
#' @keywords internal
make_optimizer <- function(params, defaults, step_fn) {
  if (is.list(params)) {
    params <- Filter(function(p) inherits(p, "torch_tensor"), params)
  }
  state <- new.env(parent = emptyenv())
  opt <- list(
    param_list = params,
    defaults = defaults,
    state = state,
    .step_fn = step_fn,
    .step_count = 0L
  )
  structure(opt, class = "torch_optimizer")
}

# Extend $ dispatch for pure-R optimizers
# The existing $.torch_optimizer in optim.R handles C++-backed optimizers.
# Pure-R ones store .step_fn instead of ptr. The $ method checks which.

# ---- RMSprop ----

#' RMSprop optimizer
#'
#' @param params List of tensors to optimize.
#' @param lr Learning rate. Default 0.01.
#' @param alpha Smoothing constant. Default 0.99.
#' @param eps Term for numerical stability. Default 1e-8.
#' @param weight_decay Weight decay (L2 penalty). Default 0.
#' @param momentum Momentum factor. Default 0.
#' @param centered If TRUE, compute centered RMSprop. Default FALSE.
#' @return A torch_optimizer object.
#' @export
optim_rmsprop <- function(params, lr = 0.01, alpha = 0.99, eps = 1e-8,
                           weight_decay = 0, momentum = 0, centered = FALSE) {
  defaults <- list(lr = lr, alpha = alpha, eps = eps,
                   weight_decay = weight_decay, momentum = momentum,
                   centered = centered)
  step_fn <- function(opt) {
    d <- opt$defaults
    for (i in seq_along(opt$param_list)) {
      p <- opt$param_list[[i]]
      g <- p$grad
      if (is.null(g)) next

      key <- as.character(i)
      if (d$weight_decay != 0) g <- g + d$weight_decay * p

      if (is.null(opt$state[[key]])) {
        opt$state[[key]] <- list(
          square_avg = torch_zeros_like(p),
          step = 0L
        )
        if (d$momentum > 0) opt$state[[key]]$momentum_buffer <- torch_zeros_like(p)
        if (d$centered) opt$state[[key]]$grad_avg <- torch_zeros_like(p)
      }
      s <- opt$state[[key]]
      s$step <- s$step + 1L

      s$square_avg$mul_(d$alpha)$add_(g * g, alpha = 1 - d$alpha)

      if (d$centered) {
        s$grad_avg$mul_(d$alpha)$add_(g, alpha = 1 - d$alpha)
        avg <- s$square_avg - s$grad_avg * s$grad_avg
      } else {
        avg <- s$square_avg
      }

      denom <- torch_sqrt(avg)$add_(d$eps)

      if (d$momentum > 0) {
        s$momentum_buffer$mul_(d$momentum)$add_(g / denom)
        p$add_(s$momentum_buffer, alpha = -d$lr)
      } else {
        p$add_(g / denom, alpha = -d$lr)
      }
      opt$state[[key]] <- s
    }
  }
  make_optimizer(params, defaults, step_fn)
}

# ---- Adagrad ----

#' Adagrad optimizer
#'
#' @param params List of tensors to optimize.
#' @param lr Learning rate. Default 0.01.
#' @param lr_decay Learning rate decay. Default 0.
#' @param weight_decay Weight decay (L2 penalty). Default 0.
#' @param eps Term for numerical stability. Default 1e-10.
#' @return A torch_optimizer object.
#' @export
optim_adagrad <- function(params, lr = 0.01, lr_decay = 0,
                           weight_decay = 0, eps = 1e-10) {
  defaults <- list(lr = lr, lr_decay = lr_decay,
                   weight_decay = weight_decay, eps = eps)
  step_fn <- function(opt) {
    d <- opt$defaults
    opt$.step_count <- opt$.step_count + 1L
    for (i in seq_along(opt$param_list)) {
      p <- opt$param_list[[i]]
      g <- p$grad
      if (is.null(g)) next

      key <- as.character(i)
      if (is.null(opt$state[[key]])) {
        opt$state[[key]] <- list(sum = torch_zeros_like(p))
      }
      s <- opt$state[[key]]

      if (d$weight_decay != 0) g <- g + d$weight_decay * p

      clr <- d$lr / (1 + (opt$.step_count - 1L) * d$lr_decay)
      s$sum$add_(g * g)
      p$add_(g / (torch_sqrt(s$sum) + d$eps), alpha = -clr)
      opt$state[[key]] <- s
    }
  }
  make_optimizer(params, defaults, step_fn)
}

# ---- Adadelta ----

#' Adadelta optimizer
#'
#' @param params List of tensors to optimize.
#' @param lr Learning rate. Default 1.0.
#' @param rho Decay rate. Default 0.9.
#' @param eps Term for numerical stability. Default 1e-6.
#' @param weight_decay Weight decay (L2 penalty). Default 0.
#' @return A torch_optimizer object.
#' @export
optim_adadelta <- function(params, lr = 1.0, rho = 0.9, eps = 1e-6,
                            weight_decay = 0) {
  defaults <- list(lr = lr, rho = rho, eps = eps, weight_decay = weight_decay)
  step_fn <- function(opt) {
    d <- opt$defaults
    for (i in seq_along(opt$param_list)) {
      p <- opt$param_list[[i]]
      g <- p$grad
      if (is.null(g)) next

      key <- as.character(i)
      if (is.null(opt$state[[key]])) {
        opt$state[[key]] <- list(
          square_avg = torch_zeros_like(p),
          acc_delta = torch_zeros_like(p)
        )
      }
      s <- opt$state[[key]]

      if (d$weight_decay != 0) g <- g + d$weight_decay * p

      s$square_avg$mul_(d$rho)$add_(g * g, alpha = 1 - d$rho)
      std <- torch_sqrt(s$square_avg + d$eps)
      delta <- torch_sqrt(s$acc_delta + d$eps) / std * g
      p$add_(delta, alpha = -d$lr)
      s$acc_delta$mul_(d$rho)$add_(delta * delta, alpha = 1 - d$rho)
      opt$state[[key]] <- s
    }
  }
  make_optimizer(params, defaults, step_fn)
}

# ---- ASGD ----

#' Averaged SGD optimizer
#'
#' @param params List of tensors to optimize.
#' @param lr Learning rate. Default 0.01.
#' @param lambd Decay term. Default 1e-4.
#' @param alpha Power for eta update. Default 0.75.
#' @param t0 Point at which to start averaging. Default 1e6.
#' @param weight_decay Weight decay (L2 penalty). Default 0.
#' @return A torch_optimizer object.
#' @export
optim_asgd <- function(params, lr = 0.01, lambd = 1e-4, alpha = 0.75,
                        t0 = 1e6, weight_decay = 0) {
  defaults <- list(lr = lr, lambd = lambd, alpha = alpha,
                   t0 = t0, weight_decay = weight_decay)
  step_fn <- function(opt) {
    d <- opt$defaults
    opt$.step_count <- opt$.step_count + 1L
    for (i in seq_along(opt$param_list)) {
      p <- opt$param_list[[i]]
      g <- p$grad
      if (is.null(g)) next

      key <- as.character(i)
      if (is.null(opt$state[[key]])) {
        opt$state[[key]] <- list(
          eta = d$lr,
          mu = 1,
          ax = torch_zeros_like(p)
        )
      }
      s <- opt$state[[key]]

      if (d$weight_decay != 0) g <- g + d$weight_decay * p

      p$add_(g, alpha = -s$eta)

      # Update ax (averaged parameters)
      if (s$mu != 1) {
        s$ax$add_((p - s$ax) * s$mu)
      } else {
        s$ax$copy_(p)
      }

      new_eta <- d$lr / (1 + d$lambd * d$lr * opt$.step_count)^d$alpha
      s$mu <- 1 / max(1, opt$.step_count - d$t0)
      s$eta <- new_eta
      opt$state[[key]] <- s
    }
  }
  make_optimizer(params, defaults, step_fn)
}

# ---- Rprop ----

#' Rprop optimizer
#'
#' @param params List of tensors to optimize.
#' @param lr Learning rate. Default 0.01.
#' @param etas Multiplicative increase/decrease factors. Default c(0.5, 1.2).
#' @param step_sizes Min/max step sizes. Default c(1e-6, 50).
#' @return A torch_optimizer object.
#' @export
optim_rprop <- function(params, lr = 0.01, etas = c(0.5, 1.2),
                         step_sizes = c(1e-6, 50)) {
  defaults <- list(lr = lr, eta_minus = etas[1], eta_plus = etas[2],
                   step_size_min = step_sizes[1], step_size_max = step_sizes[2])
  step_fn <- function(opt) {
    d <- opt$defaults
    for (i in seq_along(opt$param_list)) {
      p <- opt$param_list[[i]]
      g <- p$grad
      if (is.null(g)) next

      key <- as.character(i)
      if (is.null(opt$state[[key]])) {
        opt$state[[key]] <- list(
          prev = torch_zeros_like(g),
          step_size = torch_ones_like(g)$mul_(d$lr)
        )
      }
      s <- opt$state[[key]]

      sign_prod <- s$prev * g
      # Where sign changed: decrease step, zero prev
      neg_mask <- sign_prod$lt(0)
      # Where sign same: increase step
      pos_mask <- sign_prod$gt(0)

      new_step <- s$step_size$clone()
      new_step$masked_fill_(pos_mask, 0)$add_(
        s$step_size$mul(d$eta_plus)$masked_fill_(neg_mask$logical_not(), 0))
      new_step$masked_fill_(neg_mask, 0)$add_(
        s$step_size$mul(d$eta_minus)$masked_fill_(pos_mask$logical_not(), 0))
      new_step$clamp_(d$step_size_min, d$step_size_max)

      # Zero grad where sign changed
      g$masked_fill_(neg_mask, 0)

      p$add_(torch_sign(g) * new_step, alpha = -1)

      s$prev$copy_(g)
      s$step_size$copy_(new_step)
      opt$state[[key]] <- s
    }
  }
  make_optimizer(params, defaults, step_fn)
}

# ---- Sentinel ----

#' Required optimizer parameter sentinel
#' @export
optim_required <- structure(list(), class = "optim_required")
