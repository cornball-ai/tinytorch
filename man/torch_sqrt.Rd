% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_sqrt}
\alias{torch_sqrt}
\title{Square root}
\usage{
torch_sqrt(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Square root
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_sqrt(a)
}
}
}
