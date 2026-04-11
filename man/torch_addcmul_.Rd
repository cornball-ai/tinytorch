% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_addcmul_}
\alias{torch_addcmul_}
\title{Addcmul}
\usage{
torch_addcmul_(self, tensor1, tensor2, value = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{tensor1}{A torch_tensor.}

\item{tensor2}{A torch_tensor.}

\item{value}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::addcmul_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
