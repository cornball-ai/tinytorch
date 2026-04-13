% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_bce_loss}
\alias{nn_bce_loss}
\title{Nn bce loss}
\usage{
nn_bce_loss(weight = NULL, reduction = 1L)
}
\arguments{
\item{weight}{A `torch_tensor` of weights.}

\item{reduction}{Parameter passed to the underlying ATen operator.}
}
\value{
An `nn_module` instance.
}
\description{
Nn bce loss
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
