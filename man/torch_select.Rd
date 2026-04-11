% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_select}
\alias{torch_select}
\title{Select}
\usage{
torch_select(self, dim, index)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{A dimension name.}

\item{index}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::select}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
