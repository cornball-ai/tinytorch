% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_set_}
\alias{torch_set_}
\title{Set}
\usage{
torch_set_(self, source, storage_offset, size, stride = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{source}{A torch_tensor.}

\item{storage_offset}{An integer.}

\item{size}{An integer vector.}

\item{stride}{An integer vector.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::set_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
