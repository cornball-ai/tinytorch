% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linear}
\alias{torch_linear}
\title{Linear transformation}
\usage{
torch_linear(input, weight, bias = NULL)
}
\arguments{
\item{input}{Input tensor.}

\item{weight}{Weight matrix.}

\item{bias}{Optional bias vector.}
}
\value{
A `torch_tensor`.
}
\description{
Linear transformation
}
\examples{
\donttest{
if (torch_is_installed()) {
x <- torch_randn(c(2, 3))
w <- torch_randn(c(4, 3))
torch_linear(x, w)
}
}
}
