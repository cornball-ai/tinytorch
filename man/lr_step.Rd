% tinyrox says don't edit this manually, but it can't stop you!
\name{lr_step}
\alias{lr_step}
\title{Lr step}
\usage{
lr_step(optimizer, step_size, gamma = 0.1)
}
\arguments{
\item{optimizer}{Parameter passed to the underlying ATen operator.}

\item{step_size}{Parameter passed to the underlying ATen operator.}

\item{gamma}{Parameter passed to the underlying ATen operator.}
}
\value{
A learning rate scheduler.
}
\description{
Lr step
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
