% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_cosine_similarity}
\alias{torch_cosine_similarity}
\title{Cosine similarity}
\usage{
torch_cosine_similarity(x1, x2, dim = 1, eps = 1e-08)
}
\arguments{
\item{x1}{A torch_tensor.}

\item{x2}{A torch_tensor.}

\item{dim}{An integer.}

\item{eps}{A numeric value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::cosine_similarity}.
}
\examples{
\donttest{
if (torch_is_installed()) {

input1 = torch_randn(c(100, 128))
input2 = torch_randn(c(100, 128))
output = torch_cosine_similarity(input1, input2)
output
}
}
}
