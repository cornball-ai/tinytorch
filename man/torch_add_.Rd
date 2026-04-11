% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_add_}
\alias{torch_add_}
\title{Add}
\usage{
torch_add_(self, other, alpha = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}

\item{alpha}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::add_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
