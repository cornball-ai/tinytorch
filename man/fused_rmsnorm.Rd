% tinyrox says don't edit this manually, but it can't stop you!
\name{fused_rmsnorm}
\alias{fused_rmsnorm}
\title{Fused RMSNorm: x * rsqrt(mean(x^2) + eps) * weight}
\usage{
fused_rmsnorm(x, weight, eps = 1e-06)
}
\arguments{
\item{x}{A torch tensor.}

\item{weight}{Weight tensor (same size as last dim of x).}

\item{eps}{Small constant for numerical stability.}
}
\value{
A new torch tensor (float32).
}
\description{
Fused RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
}
\examples{
\donttest{
if (torch_is_installed()) {
x <- torch_randn(c(2, 4))
w <- torch_ones(c(4))
fused_rmsnorm(x, w)
}
}
}
