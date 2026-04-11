% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_lcm}
\alias{torch_lcm}
\title{Lcm}
\usage{
torch_lcm(self, other)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::lcm}.
}
\examples{
\donttest{
if (torch_is_installed()) {

if (cuda_is_available()) {
a <- torch_tensor(c(5, 10, 15), dtype = torch_long(), device = "cuda")
b <- torch_tensor(c(3, 4, 5), dtype = torch_long(), device = "cuda")
torch_lcm(a, b)
c <- torch_tensor(c(3L), device = "cuda")
torch_lcm(a, c)
}
}
}
}
