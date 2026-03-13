% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_addmm}
\alias{torch_addmm}
\title{Addmm}
\usage{
torch_addmm(self, mat1, mat2, beta = 1, alpha = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{mat1}{A torch_tensor.}

\item{mat2}{A torch_tensor.}

\item{beta}{A numeric scalar.}

\item{alpha}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::addmm}.
}
