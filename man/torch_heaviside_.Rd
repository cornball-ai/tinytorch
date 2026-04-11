% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_heaviside_}
\alias{torch_heaviside_}
\title{Heaviside}
\usage{
torch_heaviside_(self, values)
}
\arguments{
\item{self}{A torch_tensor.}

\item{values}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::heaviside_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
