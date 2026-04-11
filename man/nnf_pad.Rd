% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_pad}
\alias{nnf_pad}
\title{Pad a tensor}
\usage{
nnf_pad(input, pad, mode = "constant", value = 0)
}
\arguments{
\item{input}{A torch_tensor.}

\item{pad}{Integer vector of padding sizes.}

\item{mode}{Padding mode: "constant", "reflect", "replicate", or "circular".
Default "constant".}

\item{value}{Fill value for constant padding. Default 0.}
}
\value{
A torch_tensor.
}
\description{
Pads the input tensor using the specified padding mode.
}
\examples{
\donttest{
if (torch_is_installed()) {
x <- torch_randn(c(2, 3))
nnf_pad(x, c(1L, 1L))
}
}
}
