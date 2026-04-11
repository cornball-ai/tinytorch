% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_diag}
\alias{torch_diag}
\title{Diag}
\usage{
torch_diag(self, diagonal = 0)
}
\arguments{
\item{self}{A torch_tensor.}

\item{diagonal}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::diag}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
