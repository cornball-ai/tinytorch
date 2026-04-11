# Probability distributions
#
# Environment-based distribution objects using tensor ops.
# Adapted from torch R package (MIT, Daniel Falbel).

# ---- Distribution base ----

#' @keywords internal
make_distribution <- function(class, ...) {
  self <- new.env(parent = emptyenv())
  self$.class <- class
  args <- list(...)
  for (nm in names(args)) self[[nm]] <- args[[nm]]
  class(self) <- c(class, "torch_distribution")
  self
}

#' @export
print.torch_distribution <- function(x, ...) {
  cat(x$.class, "distribution\n")
  invisible(x)
}

# ---- Normal ----

#' Normal distribution
#'
#' @param loc Mean (tensor or numeric).
#' @param scale Standard deviation (tensor or numeric).
#' @return A distribution object with sample(), log_prob(), entropy() methods.
#' @export
distr_normal <- function(loc, scale) {
  if (!inherits(loc, "torch_tensor")) loc <- torch_tensor(loc)
  if (!inherits(scale, "torch_tensor")) scale <- torch_tensor(scale)
  d <- make_distribution("Normal", loc = loc, scale = scale)
  d$sample <- function(sample_shape = NULL) {
    shape <- if (is.null(sample_shape)) d$loc$size() else c(sample_shape, d$loc$size())
    torch_randn(shape) * d$scale + d$loc
  }
  d$log_prob <- function(value) {
    var <- d$scale * d$scale
    -((value - d$loc)^2) / (2 * var) - torch_log(d$scale) - 0.5 * log(2 * pi)
  }
  d$entropy <- function() {
    0.5 + 0.5 * log(2 * pi) + torch_log(d$scale)
  }
  d$mean <- d$loc
  d$variance <- d$scale^2
  d
}

# ---- Bernoulli ----

#' Bernoulli distribution
#'
#' @param probs Probability of 1 (tensor or numeric). One of probs or logits required.
#' @param logits Log-odds (tensor or numeric).
#' @return A distribution object.
#' @export
distr_bernoulli <- function(probs = NULL, logits = NULL) {
  if (is.null(probs) && is.null(logits)) stop("One of probs or logits required")
  if (!is.null(probs) && !inherits(probs, "torch_tensor")) probs <- torch_tensor(probs)
  if (!is.null(logits) && !inherits(logits, "torch_tensor")) logits <- torch_tensor(logits)
  if (is.null(probs)) probs <- torch_sigmoid(logits)
  if (is.null(logits)) logits <- torch_log(probs / (1 - probs))
  d <- make_distribution("Bernoulli", probs = probs, logits = logits)
  d$sample <- function(sample_shape = NULL) {
    shape <- if (is.null(sample_shape)) d$probs$size() else c(sample_shape, d$probs$size())
    torch_zeros(shape)$bernoulli_(d$probs)
  }
  d$log_prob <- function(value) {
    -(torch_relu(-d$logits) + torch_log(1 + torch_exp(-torch_abs(d$logits))) -
      value * d$logits)
  }
  d$entropy <- function() {
    -(d$logits * (d$probs - 1) - torch_log(1 + torch_exp(-torch_abs(d$logits))))
  }
  d$mean <- d$probs
  d
}

# ---- Categorical ----

#' Categorical distribution
#'
#' @param probs Category probabilities (tensor).
#' @param logits Unnormalized log probabilities (tensor).
#' @return A distribution object.
#' @export
distr_categorical <- function(probs = NULL, logits = NULL) {
  if (is.null(probs) && is.null(logits)) stop("One of probs or logits required")
  if (!is.null(probs) && !inherits(probs, "torch_tensor")) probs <- torch_tensor(probs)
  if (!is.null(logits) && !inherits(logits, "torch_tensor")) logits <- torch_tensor(logits)
  if (is.null(logits)) logits <- torch_log(probs)
  if (is.null(probs)) probs <- torch_softmax(logits, dim = -1L)
  d <- make_distribution("Categorical", probs = probs, logits = logits)
  d$sample <- function(sample_shape = NULL) {
    torch_multinomial(d$probs, num_samples = 1L, replacement = TRUE)$squeeze(-1L)
  }
  d$log_prob <- function(value) {
    log_probs <- torch_log_softmax(d$logits, dim = -1L)
    value <- value$to(dtype = torch_long)
    torch_gather(log_probs, -1L, value$unsqueeze(-1L))$squeeze(-1L)
  }
  d$entropy <- function() {
    p_log_p <- d$logits * d$probs
    -(p_log_p$sum(dim = -1L))
  }
  d
}

# ---- Poisson ----

#' Poisson distribution
#'
#' @param rate Rate parameter (tensor or numeric).
#' @return A distribution object.
#' @export
distr_poisson <- function(rate) {
  if (!inherits(rate, "torch_tensor")) rate <- torch_tensor(rate)
  d <- make_distribution("Poisson", rate = rate)
  d$sample <- function(sample_shape = NULL) {
    torch_poisson(d$rate)
  }
  d$log_prob <- function(value) {
    value * torch_log(d$rate) - d$rate - torch_lgamma(value + 1)
  }
  d$mean <- d$rate
  d$variance <- d$rate
  d
}

# ---- Gamma ----

#' Gamma distribution
#'
#' @param concentration Shape parameter (tensor or numeric).
#' @param rate Rate parameter (tensor or numeric).
#' @return A distribution object.
#' @export
distr_gamma <- function(concentration, rate) {
  if (!inherits(concentration, "torch_tensor")) concentration <- torch_tensor(concentration)
  if (!inherits(rate, "torch_tensor")) rate <- torch_tensor(rate)
  d <- make_distribution("Gamma", concentration = concentration, rate = rate)
  d$sample <- function(sample_shape = NULL) {
    # Use _standard_gamma and scale
    torch_exp(torch_log(concentration) - torch_log(rate)) *
      torch_randn(concentration$size())$abs()
  }
  d$log_prob <- function(value) {
    d$concentration * torch_log(d$rate) +
      (d$concentration - 1) * torch_log(value) -
      d$rate * value -
      torch_lgamma(d$concentration)
  }
  d$mean <- d$concentration / d$rate
  d$variance <- d$concentration / (d$rate^2)
  d
}

# ---- Chi2 ----

#' Chi-squared distribution
#'
#' @param df Degrees of freedom (tensor or numeric).
#' @return A distribution object.
#' @export
distr_chi2 <- function(df) {
  distr_gamma(concentration = df / 2, rate = 0.5)
}

# ---- Multivariate Normal ----

#' Multivariate normal distribution
#'
#' @param loc Mean vector (tensor).
#' @param covariance_matrix Covariance matrix (tensor). One of covariance_matrix,
#'   precision_matrix, or scale_tril required.
#' @param precision_matrix Precision matrix (tensor).
#' @param scale_tril Lower-triangular scale matrix (tensor).
#' @return A distribution object.
#' @export
distr_multivariate_normal <- function(loc, covariance_matrix = NULL,
                                       precision_matrix = NULL,
                                       scale_tril = NULL) {
  if (!inherits(loc, "torch_tensor")) loc <- torch_tensor(loc)
  if (!is.null(covariance_matrix) && !inherits(covariance_matrix, "torch_tensor"))
    covariance_matrix <- torch_tensor(covariance_matrix)
  if (!is.null(scale_tril)) {
    # Use provided
  } else if (!is.null(covariance_matrix)) {
    scale_tril <- torch_linalg_cholesky(covariance_matrix)
  } else if (!is.null(precision_matrix)) {
    scale_tril <- torch_linalg_cholesky(torch_linalg_inv(precision_matrix))
  } else {
    stop("One of covariance_matrix, precision_matrix, or scale_tril required")
  }
  d <- make_distribution("MultivariateNormal", loc = loc, scale_tril = scale_tril)
  d$sample <- function(sample_shape = NULL) {
    k <- loc$size()[length(loc$size())]
    z <- torch_randn(c(if (!is.null(sample_shape)) sample_shape, k))
    d$loc + torch_matmul(z, d$scale_tril$t())
  }
  d$log_prob <- function(value) {
    diff <- value - d$loc
    M <- torch_sum(torch_linalg_solve_triangular(d$scale_tril, diff$unsqueeze(-1L),
                                                  upper = FALSE)^2, dim = -2L)$squeeze(-1L)
    k <- as.numeric(loc$size()[length(loc$size())])
    log_det <- 2 * torch_sum(torch_log(torch_diag(d$scale_tril)))
    -0.5 * (k * log(2 * pi) + log_det + M)
  }
  d$mean <- d$loc
  d
}

# ---- Mixture ----

#' Mixture of same-family distributions
#'
#' @param mixture_distribution Categorical distribution for mixture weights.
#' @param component_distribution Batch of component distributions.
#' @return A distribution object.
#' @export
distr_mixture_same_family <- function(mixture_distribution, component_distribution) {
  d <- make_distribution("MixtureSameFamily",
    mixture_distribution = mixture_distribution,
    component_distribution = component_distribution)
  d$log_prob <- function(value) {
    log_mix <- torch_log_softmax(d$mixture_distribution$logits, dim = -1L)
    log_comp <- d$component_distribution$log_prob(value$unsqueeze(-1L))
    torch_logsumexp(log_mix + log_comp, dim = -1L)
  }
  d
}
