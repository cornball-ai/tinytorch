% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_nll_loss}
\alias{nn_nll_loss}
\title{Nn nll loss}
\usage{
nn_nll_loss(weight = NULL, reduction = 1L, ignore_index = -100)
}
\arguments{
\item{weight}{A `torch_tensor` of weights.}

\item{reduction}{Parameter passed to the underlying ATen operator.}

\item{ignore_index}{Parameter passed to the underlying ATen operator.}
}
\value{
An `nn_module` instance.
}
\description{
Nn nll loss
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
