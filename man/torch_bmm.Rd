% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_bmm}
\alias{torch_bmm}
\title{torch_bmm}
\usage{
torch_bmm(self, mat2)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{mat2}{A `torch_tensor` (matrix).}
}
\value{
A `torch_tensor`.
}
\description{
torch_bmm
}
\examples{
\donttest{
if (torch_is_installed()) {

input = torch_randn(c(10, 3, 4))
mat2 = torch_randn(c(10, 4, 5))
res = torch_bmm(input, mat2)
res
}
}
}
