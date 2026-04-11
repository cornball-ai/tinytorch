% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_all}
\alias{torch_all}
\title{All}
\usage{
torch_all(self, dim, keepdim = FALSE)
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
Wrapper for \code{at::all}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
