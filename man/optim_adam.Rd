% tinyrox says don't edit this manually, but it can't stop you!
\name{optim_adam}
\alias{optim_adam}
\title{Adam optimizer}
\usage{
optim_adam(params, lr = 0.001, betas = c(0.9, 0.999), eps = 1e-08,
           weight_decay = 0, amsgrad = FALSE)
}
\arguments{
\item{params}{List of tensors to optimize.}

\item{lr}{Learning rate. Default 0.001.}

\item{betas}{Coefficients for computing running averages. Default c(0.9, 0.999).}

\item{eps}{Term for numerical stability. Default 1e-8.}

\item{weight_decay}{Weight decay (L2 penalty). Default 0.}

\item{amsgrad}{Whether to use AMSGrad variant. Default FALSE.}
}
\value{
A torch_optimizer object.
}
\description{
Adam optimizer
}
\examples{
\donttest{
if (torch_is_installed()) {
p <- nn_parameter(torch_randn(5))
opt <- optim_adam(list(p), lr = 0.001)
loss <- (p ^ 2)$sum()
loss$backward()
opt$step()
opt$zero_grad()
}
}
}
