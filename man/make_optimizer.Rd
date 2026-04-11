% tinyrox says don't edit this manually, but it can't stop you!
\name{make_optimizer}
\alias{make_optimizer}
\title{Create a pure-R optimizer}
\usage{
make_optimizer(params, defaults, step_fn)
}
\arguments{
\item{params}{List of tensors to optimize (must have requires_grad).}

\item{defaults}{Named list of default hyperparameters.}

\item{step_fn}{Function; optimizer step callback.}
}
\value{
An optimizer environment with step() and zero_grad() methods.
}
\description{
Create a pure-R optimizer
}
\keyword{internal}
