% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_gelu_}
\alias{torch_gelu_}
\title{Gelu}
\usage{
torch_gelu_(self, approximate = "none")
}
\arguments{
\item{self}{A torch_tensor.}

\item{approximate}{A character string.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::gelu_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
