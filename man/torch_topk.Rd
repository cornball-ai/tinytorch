% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_topk}
\alias{torch_topk}
\title{Topk}
\usage{
torch_topk(self, k, dim = -1, largest = TRUE, sorted = TRUE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{k}{An integer.}

\item{dim}{An integer.}

\item{largest}{A logical value.}

\item{sorted}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::topk}.
}
\examples{
\donttest{
if (torch_is_installed()) {

x = torch_arange(1., 6.)
x
torch_topk(x, 3)
}
}
}
