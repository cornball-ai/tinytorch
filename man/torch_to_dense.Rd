% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_to_dense}
\alias{torch_to_dense}
\title{To dense}
\usage{
torch_to_dense(self, dtype = NULL, masked_grad = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{masked_grad}{A logical value. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::to_dense}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
