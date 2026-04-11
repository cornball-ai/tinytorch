% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_copy}
\alias{torch_copy}
\title{Copy}
\usage{
torch_copy(self, src, non_blocking = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{src}{A torch_tensor.}

\item{non_blocking}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::copy}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
