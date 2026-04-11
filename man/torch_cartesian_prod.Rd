% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_cartesian_prod}
\alias{torch_cartesian_prod}
\title{Cartesian prod}
\usage{
torch_cartesian_prod(tensors)
}
\arguments{
\item{tensors}{A list of torch_tensors.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::cartesian_prod}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = c(1, 2, 3)
b = c(4, 5)
tensor_a = torch_tensor(a)
tensor_b = torch_tensor(b)
torch_cartesian_prod(list(tensor_a, tensor_b))
}
}
}
