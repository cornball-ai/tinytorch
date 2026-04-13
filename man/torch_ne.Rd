% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_ne}
\alias{torch_ne}
\title{torch_ne}
\usage{
torch_ne(self, other)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{other}{A `torch_tensor` or scalar.}
}
\value{
A `torch_tensor`.
}
\description{
torch_ne
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_ne(torch_tensor(matrix(1:4, ncol = 2, byrow=TRUE)), 
         torch_tensor(matrix(rep(c(1,4), each = 2), ncol = 2, byrow=TRUE)))
}
}
}
