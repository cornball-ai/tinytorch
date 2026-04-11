% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_linear}
\alias{nn_linear}
\title{Linear layer}
\usage{
nn_linear(in_features, out_features, bias = TRUE)
}
\arguments{
\item{in_features}{Integer, size of each input sample.}

\item{out_features}{Integer, size of each output sample.}

\item{bias}{Logical, whether to include a bias term.}
}
\value{
An nn_module instance.
}
\description{
Linear layer
}
\examples{
\donttest{
if (torch_is_installed()) {
m <- nn_linear(3, 2)
m(torch_randn(c(1, 3)))
}
}
}
