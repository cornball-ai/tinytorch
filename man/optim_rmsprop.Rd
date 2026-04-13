% tinyrox says don't edit this manually, but it can't stop you!
\name{optim_rmsprop}
\alias{optim_rmsprop}
\title{RMSprop optimizer}
\usage{
optim_rmsprop(params, lr = 0.01, alpha = 0.99, eps = 1e-08, weight_decay = 0,
              momentum = 0, centered = FALSE)
}
\arguments{
\item{params}{List of tensors to optimize.}

\item{lr}{Learning rate. Default 0.01.}

\item{alpha}{Smoothing constant. Default 0.99.}

\item{eps}{Term for numerical stability. Default 1e-8.}

\item{weight_decay}{Weight decay (L2 penalty). Default 0.}

\item{momentum}{Momentum factor. Default 0.}

\item{centered}{If TRUE, compute centered RMSprop. Default FALSE.}
}
\value{
A torch_optimizer object.
}
\description{
RMSprop optimizer
}
\examples{
\donttest{
if (torch_is_installed()) {
p <- nn_parameter(torch_randn(5))
opt <- optim_rmsprop(list(p), lr = 0.01)
loss <- (p ^ 2)$sum()
loss$backward()
opt$step()
opt$zero_grad()
}
}
}
