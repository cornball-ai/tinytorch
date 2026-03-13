% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_put_}
\alias{torch_put_}
\title{Put}
\usage{
torch_put_(self, index, source, accumulate = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{index}{A torch_tensor.}

\item{source}{A torch_tensor.}

\item{accumulate}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::put_}.
}
