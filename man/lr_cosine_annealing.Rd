% tinyrox says don't edit this manually, but it can't stop you!
\name{lr_cosine_annealing}
\alias{lr_cosine_annealing}
\title{Lr cosine annealing}
\usage{
lr_cosine_annealing(optimizer, T_max, eta_min = 0)
}
\arguments{
\item{optimizer}{Parameter passed to the underlying ATen operator.}

\item{T_max}{Parameter passed to the underlying ATen operator.}

\item{eta_min}{Parameter passed to the underlying ATen operator.}
}
\value{
A learning rate scheduler.
}
\description{
Lr cosine annealing
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
