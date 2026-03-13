% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_addcmul}
\alias{torch_addcmul}
\title{Addcmul}
\usage{
torch_addcmul(self, tensor1, tensor2, value = 1)
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
Wrapper for \code{at::addcmul}.
}
