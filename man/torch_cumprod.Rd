% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_cumprod}
\alias{torch_cumprod}
\title{Cumprod}
\usage{
torch_cumprod(self, dim, dtype = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{An integer.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::cumprod}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
