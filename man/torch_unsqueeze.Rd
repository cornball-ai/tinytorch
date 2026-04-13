% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_unsqueeze}
\alias{torch_unsqueeze}
\title{torch_unsqueeze}
\usage{
torch_unsqueeze(self, dim)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{dim}{Integer dimension.}
}
\value{
A `torch_tensor`.
}
\description{
torch_unsqueeze
}
\examples{
\donttest{
if (torch_is_installed()) {

x = torch_tensor(c(1, 2, 3, 4))
torch_unsqueeze(x, 1)
torch_unsqueeze(x, 2)
}
}
}
