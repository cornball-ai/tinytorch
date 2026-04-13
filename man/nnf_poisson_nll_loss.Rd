% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_poisson_nll_loss}
\alias{nnf_poisson_nll_loss}
\title{Nnf poisson nll loss}
\usage{
nnf_poisson_nll_loss(input, target, log_input, full, eps, reduction)
}
\arguments{
\item{input}{A `torch_tensor`.}

\item{target}{Parameter passed to the underlying ATen operator.}

\item{log_input}{Parameter passed to the underlying ATen operator.}

\item{full}{Parameter passed to the underlying ATen operator.}

\item{eps}{Numeric; small constant for numerical stability.}

\item{reduction}{Parameter passed to the underlying ATen operator.}
}
\value{
A `torch_tensor`.
}
\description{
Nnf poisson nll loss
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
