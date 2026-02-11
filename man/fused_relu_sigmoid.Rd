% tinyrox says don't edit this manually, but it can't stop you!
\name{fused_relu_sigmoid}
\alias{fused_relu_sigmoid}
\title{Fused relu + sigmoid}
\usage{
fused_relu_sigmoid(x)
}
\arguments{
\item{x}{A torch tensor.}
}
\value{
A new torch tensor (float32).
}
\description{
Fused relu + sigmoid
}
\examples{
\donttest{
fused_relu_sigmoid(torch_randn(c(2, 3)))
}
}
