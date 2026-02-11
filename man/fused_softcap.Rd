% tinyrox says don't edit this manually, but it can't stop you!
\name{fused_softcap}
\alias{fused_softcap}
\title{Fused softcap: tanh(x/cap) * cap}
\usage{
fused_softcap(x, cap)
}
\arguments{
\item{x}{A torch tensor.}

\item{cap}{Scalar capping value.}
}
\value{
A new torch tensor (float32).
}
\description{
Fused softcap: tanh(x/cap) * cap
}
\examples{
\donttest{
fused_softcap(torch_randn(c(2, 3)), 30.0)
}
}
