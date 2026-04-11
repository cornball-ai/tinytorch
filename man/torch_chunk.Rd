% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_chunk}
\alias{torch_chunk}
\title{Chunk}
\usage{
torch_chunk(self, chunks, dim = 0)
}
\arguments{
\item{self}{A torch_tensor.}

\item{chunks}{An integer.}

\item{dim}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::chunk}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
