% tinyrox says don't edit this manually, but it can't stop you!
\name{optim_adagrad}
\alias{optim_adagrad}
\title{Adagrad optimizer}
\usage{
optim_adagrad(params, lr = 0.01, lr_decay = 0, weight_decay = 0, eps = 1e-10)
}
\arguments{
\item{params}{List of tensors to optimize.}

\item{lr}{Learning rate. Default 0.01.}

\item{lr_decay}{Learning rate decay. Default 0.}

\item{weight_decay}{Weight decay (L2 penalty). Default 0.}

\item{eps}{Term for numerical stability. Default 1e-10.}
}
\value{
A torch_optimizer object.
}
\description{
Adagrad optimizer
}
\examples{
\donttest{
if (torch_is_installed()) {
params <- list(nn_parameter(torch_randn(3)))
opt <- optim_adagrad(params, lr = 0.01)
opt$step()
}
}
}
