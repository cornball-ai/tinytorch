% tinyrox says don't edit this manually, but it can't stop you!
\name{optim_sgd}
\alias{optim_sgd}
\title{SGD optimizer}
\usage{
optim_sgd(params, lr, momentum = 0, dampening = 0, weight_decay = 0,
          nesterov = FALSE)
}
\arguments{
\item{params}{List of tensors to optimize.}

\item{lr}{Learning rate.}

\item{momentum}{Momentum factor. Default 0.}

\item{dampening}{Dampening for momentum. Default 0.}

\item{weight_decay}{Weight decay (L2 penalty). Default 0.}

\item{nesterov}{Whether to use Nesterov momentum. Default FALSE.}
}
\value{
A torch_optimizer object.
}
\description{
SGD optimizer
}
\examples{
\donttest{
if (torch_is_installed()) {

optimizer <- optim_sgd(model$parameters(), lr = 0.1, momentum = 0.9)
optimizer$zero_grad()
loss_fn(model(input), target)$backward()
optimizer$step()


}
}
}
