% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_constant_pad_nd}
\alias{torch_constant_pad_nd}
\title{Constant pad nd}
\usage{
torch_constant_pad_nd(self, pad, value = 0)
}
\arguments{
\item{self}{A torch_tensor.}

\item{pad}{An integer vector.}

\item{value}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::constant_pad_nd}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
