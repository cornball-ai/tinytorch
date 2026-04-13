% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_huber_loss}
\alias{nn_huber_loss}
\title{Nn huber loss}
\usage{
nn_huber_loss(reduction = 1L, delta = 1)
}
\arguments{
\item{reduction}{Parameter passed to the underlying ATen operator.}

\item{delta}{Parameter passed to the underlying ATen operator.}
}
\value{
An `nn_module` instance.
}
\description{
Nn huber loss
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
