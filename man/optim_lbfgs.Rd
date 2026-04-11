% tinyrox says don't edit this manually, but it can't stop you!
\name{optim_lbfgs}
\alias{optim_lbfgs}
\title{L-BFGS optimizer (stub)}
\usage{
optim_lbfgs(params, lr = 1, max_iter = 20L, max_eval = NULL,
            tolerance_grad = 1e-07, tolerance_change = 1e-09,
            history_size = 100L, line_search_fn = NULL)
}
\arguments{
\item{params}{Parameters.}

\item{lr}{Learning rate.}

\item{max_iter}{Max iterations. Default 20.}

\item{max_eval}{Max evaluations.}

\item{tolerance_grad}{Gradient tolerance.}

\item{tolerance_change}{Parameter change tolerance.}

\item{history_size}{History size. Default 100.}

\item{line_search_fn}{Line search function.}
}
\value{
A torch_optimizer.
}
\description{
L-BFGS optimizer (stub)
}
\examples{
\donttest{
if (torch_is_installed()) {
a <- 1
b <- 5
rosenbrock <- function(x) {
  x1 <- x[1]
  x2 <- x[2]
  (a - x1)^2 + b * (x2 - x1^2)^2
}

x <- torch_tensor(c(-1, 1), requires_grad = TRUE)

optimizer <- optim_lbfgs(x)
calc_loss <- function() {
  optimizer$zero_grad()
  value <- rosenbrock(x)
  value$backward()
  value
}

num_iterations <- 2
for (i in 1:num_iterations) {
  optimizer$step(calc_loss)
}

rosenbrock(x)

}
}
}
