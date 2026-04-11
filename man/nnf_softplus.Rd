% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_softplus}
\alias{nnf_softplus}
\title{Softplus activation}
\usage{
nnf_softplus(input, beta = 1, threshold = 20)
}
\arguments{
\item{input}{A torch_tensor.}

\item{beta}{Multiplier for the input. Default 1.0.}

\item{threshold}{Values above this revert to linear. Default 20.0.}
}
\value{
A torch_tensor.
}
\description{
Applies the softplus function element-wise.
}
\examples{
\donttest{
if (torch_is_installed()) {
nnf_softplus(torch_randn(5))
}
}
}
