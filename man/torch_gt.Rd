% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_gt}
\alias{torch_gt}
\title{torch_gt}
\usage{
torch_gt(self, other)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{other}{A `torch_tensor` or scalar.}
}
\description{
torch_gt
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_gt(torch_tensor(matrix(1:4, ncol = 2, byrow=TRUE)), 
         torch_tensor(matrix(c(1,1,4,4), ncol = 2, byrow=TRUE)))
}
}
}
