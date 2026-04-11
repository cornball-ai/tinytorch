% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_concat}
\alias{torch_concat}
\title{Concat}
\usage{
torch_concat(tensors, dim = 0)
}
\arguments{
\item{tensors}{A list of torch_tensors.}

\item{dim}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::concat}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
