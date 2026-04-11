% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_histogramdd}
\alias{torch_histogramdd}
\title{Histogramdd}
\usage{
torch_histogramdd(self, bins, range = NULL, weight = NULL, density = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{bins}{An integer vector.}

\item{range}{A numeric vector. Optional.}

\item{weight}{A torch_tensor. Optional.}

\item{density}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::histogramdd}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
