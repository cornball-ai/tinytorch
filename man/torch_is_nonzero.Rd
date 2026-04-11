% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_is_nonzero}
\alias{torch_is_nonzero}
\title{Is nonzero}
\usage{
torch_is_nonzero(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::is_nonzero}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_is_nonzero(torch_tensor(c(0.)))
torch_is_nonzero(torch_tensor(c(1.5)))
torch_is_nonzero(torch_tensor(c(FALSE)))
torch_is_nonzero(torch_tensor(c(3)))
if (FALSE) {
torch_is_nonzero(torch_tensor(c(1, 3, 5)))
torch_is_nonzero(torch_tensor(c()))
}
}
}
}
