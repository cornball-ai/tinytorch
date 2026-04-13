% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_argmax}
\alias{torch_argmax}
\title{torch_argmax}
\usage{
torch_argmax(self, dim = NULL, keepdim = FALSE)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{dim}{Integer dimension.}

\item{keepdim}{Logical; keep reduced dimensions.}
}
\value{
A `torch_tensor`.
}
\description{
torch_argmax
}
\examples{
\donttest{
if (torch_is_installed()) {


a = torch_randn(c(4, 4))
a
torch_argmax(a)



a = torch_randn(c(4, 4))
a
torch_argmax(a, dim=1)
}
}
}
