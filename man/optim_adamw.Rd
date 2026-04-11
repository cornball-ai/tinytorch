% tinyrox says don't edit this manually, but it can't stop you!
\name{optim_adamw}
\alias{optim_adamw}
\title{AdamW optimizer}
\usage{
optim_adamw(params, lr = 0.001, betas = c(0.9, 0.999), eps = 1e-08,
            weight_decay = 0.01, amsgrad = FALSE)
}
\arguments{
\item{params}{List of tensors to optimize.}

\item{lr}{Learning rate. Default 0.001.}

\item{betas}{Coefficients for computing running averages. Default c(0.9, 0.999).}

\item{eps}{Term for numerical stability. Default 1e-8.}

\item{weight_decay}{Weight decay. Default 0.01.}

\item{amsgrad}{Whether to use AMSGrad variant. Default FALSE.}
}
\value{
A torch_optimizer object.
}
\description{
AdamW optimizer
}
\examples{
\donttest{
if (torch_is_installed()) {
params <- list(nn_parameter(torch_randn(3)))
opt <- optim_adamw(params, lr = 0.001)
opt$step()
}
}
}
