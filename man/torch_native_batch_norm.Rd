% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_native_batch_norm}
\alias{torch_native_batch_norm}
\title{Native batch norm}
\usage{
torch_native_batch_norm(input, weight, bias, running_mean, running_var,
                        training, momentum, eps)
}
\arguments{
\item{input}{A torch_tensor.}

\item{weight}{A torch_tensor.}

\item{bias}{A torch_tensor.}

\item{running_mean}{A torch_tensor.}

\item{running_var}{A torch_tensor.}

\item{training}{A logical value.}

\item{momentum}{A numeric value.}

\item{eps}{A numeric value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::native_batch_norm}.
}
