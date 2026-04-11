% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_floor}
\alias{torch_floor}
\title{Floor}
\usage{
torch_floor(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Floor
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_floor(a)
}
}
}
