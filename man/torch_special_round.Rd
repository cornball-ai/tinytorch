% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_special_round}
\alias{torch_special_round}
\title{Special round}
\usage{
torch_special_round(self, decimals = 0)
}
\arguments{
\item{self}{A torch_tensor.}

\item{decimals}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::special_round}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
