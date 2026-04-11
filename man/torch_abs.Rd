% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_abs}
\alias{torch_abs}
\title{Absolute value}
\usage{
torch_abs(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Absolute value
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_abs(torch_tensor(c(-1, -2, 3)))
}
}
}
