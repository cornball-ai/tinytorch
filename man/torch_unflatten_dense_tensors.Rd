% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_unflatten_dense_tensors}
\alias{torch_unflatten_dense_tensors}
\title{Unflatten dense tensors}
\usage{
torch_unflatten_dense_tensors(flat, tensors)
}
\arguments{
\item{flat}{A torch_tensor.}

\item{tensors}{A list of torch_tensors.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::unflatten_dense_tensors}.
}
