% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_einsum}
\alias{torch_einsum}
\title{Einsum}
\usage{
torch_einsum(equation, tensors, path = NULL)
}
\arguments{
\item{equation}{A character string.}

\item{tensors}{A list of torch_tensors.}

\item{path}{An integer vector. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::einsum}.
}
