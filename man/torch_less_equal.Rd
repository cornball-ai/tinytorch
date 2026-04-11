% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_less_equal}
\alias{torch_less_equal}
\title{Less equal}
\usage{
torch_less_equal(self, other)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::less_equal}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
