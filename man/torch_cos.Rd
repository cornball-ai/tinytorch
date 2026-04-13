% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_cos}
\alias{torch_cos}
\title{Compute cos of tensor}
\usage{
torch_cos(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A `torch_tensor`.
}
\description{
Compute cos of tensor
}
\examples{
\donttest{
if (torch_is_installed()) {
torch_cos(torch_tensor(c(0, 3.14159)))
}
}
}
