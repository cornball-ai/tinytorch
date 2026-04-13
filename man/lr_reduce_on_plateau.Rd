% tinyrox says don't edit this manually, but it can't stop you!
\name{lr_reduce_on_plateau}
\alias{lr_reduce_on_plateau}
\title{Lr reduce on plateau}
\usage{
lr_reduce_on_plateau(optimizer, mode = "min", factor = 0.1, patience = 10)
}
\arguments{
\item{optimizer}{Parameter passed to the underlying ATen operator.}

\item{mode}{Parameter passed to the underlying ATen operator.}

\item{factor}{Parameter passed to the underlying ATen operator.}

\item{patience}{Parameter passed to the underlying ATen operator.}
}
\value{
A learning rate scheduler.
}
\description{
Lr reduce on plateau
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
