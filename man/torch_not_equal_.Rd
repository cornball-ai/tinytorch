% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_not_equal_}
\alias{torch_not_equal_}
\title{Not equal}
\usage{
torch_not_equal_(self, other)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::not_equal_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
