% tinyrox says don't edit this manually, but it can't stop you!
\name{lr_multiplicative}
\alias{lr_multiplicative}
\title{Lr multiplicative}
\usage{
lr_multiplicative(optimizer, lr_lambda)
}
\arguments{
\item{optimizer}{Parameter passed to the underlying ATen operator.}

\item{lr_lambda}{Parameter passed to the underlying ATen operator.}
}
\value{
A learning rate scheduler.
}
\description{
Lr multiplicative
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
