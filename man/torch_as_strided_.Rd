% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_as_strided_}
\alias{torch_as_strided_}
\title{As strided}
\usage{
torch_as_strided_(self, size, stride, storage_offset = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{size}{An integer vector.}

\item{stride}{An integer vector.}

\item{storage_offset}{An integer. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::as_strided_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
