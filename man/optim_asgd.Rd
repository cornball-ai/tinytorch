% tinyrox says don't edit this manually, but it can't stop you!
\name{optim_asgd}
\alias{optim_asgd}
\title{Averaged SGD optimizer}
\usage{
optim_asgd(params, lr = 0.01, lambd = 1e-04, alpha = 0.75, t0 = 1e+06,
           weight_decay = 0)
}
\arguments{
\item{params}{List of tensors to optimize.}

\item{lr}{Learning rate. Default 0.01.}

\item{lambd}{Decay term. Default 1e-4.}

\item{alpha}{Power for eta update. Default 0.75.}

\item{t0}{Point at which to start averaging. Default 1e6.}

\item{weight_decay}{Weight decay (L2 penalty). Default 0.}
}
\value{
A torch_optimizer object.
}
\description{
Averaged SGD optimizer
}
