% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_eq}
\alias{torch_eq}
\title{torch_eq}
\usage{
torch_eq(self, other)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{other}{A `torch_tensor` or scalar.}
}
\value{
A `torch_tensor`.
}
\description{
torch_eq
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_eq(torch_tensor(c(1,2,3,4)), torch_tensor(c(1, 3, 2, 4)))
}
}
}
