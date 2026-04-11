% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_histogram}
\alias{torch_histogram}
\title{Histogram}
\usage{
torch_histogram(self, bins, weight = NULL, density = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{bins}{A torch_tensor.}

\item{weight}{A torch_tensor. Optional.}

\item{density}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::histogram}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
