% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_diag_embed}
\alias{torch_diag_embed}
\title{Diag embed}
\usage{
torch_diag_embed(self, offset = 0, dim1 = -2, dim2 = -1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{offset}{An integer.}

\item{dim1}{An integer.}

\item{dim2}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::diag_embed}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(2, 3))
torch_diag_embed(a)
torch_diag_embed(a, offset=1, dim1=1, dim2=3)
}
}
}
