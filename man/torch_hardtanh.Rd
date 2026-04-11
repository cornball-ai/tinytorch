% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_hardtanh}
\alias{torch_hardtanh}
\title{Hardtanh}
\usage{
torch_hardtanh(self, min_val = -1, max_val = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{min_val}{A numeric scalar.}

\item{max_val}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::hardtanh}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
