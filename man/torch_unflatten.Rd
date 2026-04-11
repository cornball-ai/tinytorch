% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_unflatten}
\alias{torch_unflatten}
\title{Unflatten}
\usage{
torch_unflatten(self, dim, sizes)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{An integer.}

\item{sizes}{An integer vector.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::unflatten}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
