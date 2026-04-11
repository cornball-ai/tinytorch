% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_is_same_size}
\alias{torch_is_same_size}
\title{Is same size}
\usage{
torch_is_same_size(self, other)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::is_same_size}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
