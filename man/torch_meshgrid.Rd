% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_meshgrid}
\alias{torch_meshgrid}
\title{Meshgrid}
\usage{
torch_meshgrid(tensors)
}
\arguments{
\item{tensors}{A list of torch_tensors.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::meshgrid}.
}
\examples{
\donttest{
if (torch_is_installed()) {

x = torch_tensor(c(1, 2, 3))
y = torch_tensor(c(4, 5, 6))
out = torch_meshgrid(list(x, y))
out
}
}
}
