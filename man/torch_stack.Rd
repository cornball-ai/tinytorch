% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_stack}
\alias{torch_stack}
\title{Stack}
\usage{
torch_stack(tensors, dim = 0)
}
\arguments{
\item{tensors}{A list of torch_tensors.}

\item{dim}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::stack}.
}
