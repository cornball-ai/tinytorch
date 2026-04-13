% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_roll}
\alias{torch_roll}
\title{Roll}
\usage{
torch_roll(self, shifts, dims = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{shifts}{(int[1]).}

\item{dims}{(int[1]).}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::roll}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
