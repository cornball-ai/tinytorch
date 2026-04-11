% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_any}
\alias{torch_any}
\title{Any}
\usage{
torch_any(self, dim, keepdim = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{An integer.}

\item{keepdim}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::any}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
