% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_concatenate}
\alias{torch_concatenate}
\title{Concatenate}
\usage{
torch_concatenate(tensors, dim = 0)
}
\arguments{
\item{tensors}{A list of torch_tensors.}

\item{dim}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::concatenate}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
