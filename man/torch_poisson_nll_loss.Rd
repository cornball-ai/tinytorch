% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_poisson_nll_loss}
\alias{torch_poisson_nll_loss}
\title{Poisson nll loss}
\usage{
torch_poisson_nll_loss(input, target, log_input, full, eps, reduction)
}
\arguments{
\item{input}{A torch_tensor.}

\item{target}{A torch_tensor.}

\item{log_input}{A logical value.}

\item{full}{A logical value.}

\item{eps}{A numeric value.}

\item{reduction}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::poisson_nll_loss}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
