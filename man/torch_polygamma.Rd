% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_polygamma}
\alias{torch_polygamma}
\title{Polygamma}
\usage{
torch_polygamma(n, self)
}
\arguments{
\item{n}{An integer.}

\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::polygamma}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_tensor(c(1, 0.5))
torch_polygamma(1, a)

}
}
}
