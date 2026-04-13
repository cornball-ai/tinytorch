% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_poisson_nll_loss}
\alias{nn_poisson_nll_loss}
\title{Nn poisson nll loss}
\usage{
nn_poisson_nll_loss(log_input = TRUE, full = FALSE, eps = 1e-08, reduction = 1L)
}
\arguments{
\item{log_input}{Parameter passed to the underlying ATen operator.}

\item{full}{Parameter passed to the underlying ATen operator.}

\item{eps}{Numeric; small constant for numerical stability.}

\item{reduction}{Parameter passed to the underlying ATen operator.}
}
\value{
An `nn_module` instance.
}
\description{
Nn poisson nll loss
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
