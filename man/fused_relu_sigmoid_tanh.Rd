% tinyrox says don't edit this manually, but it can't stop you!
\name{fused_relu_sigmoid_tanh}
\alias{fused_relu_sigmoid_tanh}
\title{Fused relu + sigmoid + tanh}
\usage{
fused_relu_sigmoid_tanh(x)
}
\arguments{
\item{x}{A torch tensor.}
}
\value{
A new torch tensor (float32).
}
\description{
Fused relu + sigmoid + tanh
}
\examples{
\dontrun{
fused_relu_sigmoid_tanh(torch_randn(c(2, 3)))
}
}
