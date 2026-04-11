% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_sub_}
\alias{torch_sub_}
\title{Sub}
\usage{
torch_sub_(self, other, alpha = 1)
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
Wrapper for \code{at::sub_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
