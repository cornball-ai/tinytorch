% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_tril_}
\alias{torch_tril_}
\title{Tril}
\usage{
torch_tril_(self, diagonal = 0)
}
\arguments{
\item{self}{A torch_tensor.}

\item{diagonal}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::tril_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
