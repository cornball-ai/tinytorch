% tinyrox says don't edit this manually, but it can't stop you!
\name{fused_silu}
\alias{fused_silu}
\title{Fused SiLU (Swish): x * sigmoid(x)}
\usage{
fused_silu(x)
}
\arguments{
\item{x}{A torch tensor.}
}
\value{
A new torch tensor (float32).
}
\description{
Fused SiLU (Swish): x * sigmoid(x)
}
\examples{
\donttest{
fused_silu(torch_randn(c(2, 3)))
}
}
