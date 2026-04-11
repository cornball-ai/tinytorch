% tinyrox says don't edit this manually, but it can't stop you!
\name{optim_adadelta}
\alias{optim_adadelta}
\title{Adadelta optimizer}
\usage{
optim_adadelta(params, lr = 1, rho = 0.9, eps = 1e-06, weight_decay = 0)
}
\arguments{
\item{params}{List of tensors to optimize.}

\item{lr}{Learning rate. Default 1.0.}

\item{rho}{Decay rate. Default 0.9.}

\item{eps}{Term for numerical stability. Default 1e-6.}

\item{weight_decay}{Weight decay (L2 penalty). Default 0.}
}
\value{
A torch_optimizer object.
}
\description{
Adadelta optimizer
}
