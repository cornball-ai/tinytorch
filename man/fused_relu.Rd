% tinyrox says don't edit this manually, but it can't stop you!
\name{fused_relu}
\alias{fused_relu}
\title{Fused relu}
\usage{
fused_relu(x)
}
\arguments{
\item{x}{A torch tensor.}
}
\value{
A new torch tensor (float32).
}
\description{
Fused relu
}
\examples{
\donttest{
fused_relu(torch_randn(c(2, 3)))
}
}
