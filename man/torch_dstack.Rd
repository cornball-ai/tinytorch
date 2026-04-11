% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_dstack}
\alias{torch_dstack}
\title{Dstack}
\usage{
torch_dstack(tensors)
}
\arguments{
\item{tensors}{A list of torch_tensors.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::dstack}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a <- torch_tensor(c(1, 2, 3))
b <- torch_tensor(c(4, 5, 6))
torch_dstack(list(a,b))
a <- torch_tensor(rbind(1,2,3))
b <- torch_tensor(rbind(4,5,6))
torch_dstack(list(a,b))
}
}
}
