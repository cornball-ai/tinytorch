% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_logical_not}
\alias{torch_logical_not}
\title{torch_logical_not}
\usage{
torch_logical_not(self)
}
\arguments{
\item{self}{A `torch_tensor`.}
}
\value{
A `torch_tensor`.
}
\description{
torch_logical_not
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_logical_not(torch_tensor(c(TRUE, FALSE)))
torch_logical_not(torch_tensor(c(0, 1, -10), dtype=torch_int8()))
torch_logical_not(torch_tensor(c(0., 1.5, -10.), dtype=torch_double()))
}
}
}
