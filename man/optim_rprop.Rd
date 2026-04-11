% tinyrox says don't edit this manually, but it can't stop you!
\name{optim_rprop}
\alias{optim_rprop}
\title{Rprop optimizer}
\usage{
optim_rprop(params, lr = 0.01, etas = c(0.5, 1.2), step_sizes = c(1e-06, 50))
}
\arguments{
\item{params}{List of tensors to optimize.}

\item{lr}{Learning rate. Default 0.01.}

\item{etas}{Multiplicative increase/decrease factors. Default c(0.5, 1.2).}

\item{step_sizes}{Min/max step sizes. Default c(1e-6, 50).}
}
\value{
A torch_optimizer object.
}
\description{
Rprop optimizer
}
