% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_split}
\alias{torch_split}
\title{Split}
\usage{
torch_split(self, split_size, dim = 0)
}
\arguments{
\item{self}{A torch_tensor.}

\item{split_size}{An integer.}

\item{dim}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::split}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
