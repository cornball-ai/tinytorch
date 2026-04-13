% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_lt}
\alias{torch_lt}
\title{torch_lt}
\usage{
torch_lt(self, other)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{other}{A `torch_tensor` or scalar.}
}
\value{
A `torch_tensor`.
}
\description{
torch_lt
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_lt(torch_tensor(matrix(1:4, ncol = 2, byrow=TRUE)), 
         torch_tensor(matrix(c(1,1,4,4), ncol = 2, byrow=TRUE)))
}
}
}
