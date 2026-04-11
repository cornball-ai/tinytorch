% tinyrox says don't edit this manually, but it can't stop you!
\name{fused_sincos}
\alias{fused_sincos}
\title{Fused sin and cos}
\usage{
fused_sincos(x)
}
\arguments{
\item{x}{A torch tensor.}
}
\value{
A list with sin and cos tensors (float32).
}
\description{
Fused sin and cos
}
\examples{
\donttest{
if (torch_is_installed()) {
fused_sincos(torch_randn(c(2, 3)))
}
}
}
