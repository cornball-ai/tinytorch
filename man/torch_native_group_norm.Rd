% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_native_group_norm}
\alias{torch_native_group_norm}
\title{Native group norm}
\usage{
torch_native_group_norm(input, weight, bias, N, C, HxW, group, eps)
}
\arguments{
\item{input}{A torch_tensor.}

\item{weight}{A torch_tensor.}

\item{bias}{A torch_tensor.}

\item{N}{An integer.}

\item{C}{An integer.}

\item{HxW}{An integer.}

\item{group}{An integer.}

\item{eps}{A numeric value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::native_group_norm}.
}
