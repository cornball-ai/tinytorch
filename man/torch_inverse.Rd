% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_inverse}
\alias{torch_inverse}
\title{Inverse}
\usage{
torch_inverse(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::inverse}.
}
\examples{
\donttest{
if (torch_is_installed()) {

x = torch_rand(c(4, 4))
y = torch_inverse(x)
z = torch_mm(x, y)
z
torch_max(torch_abs(z - torch_eye(4))) # Max non-zero
# Batched inverse example
x = torch_randn(c(2, 3, 4, 4))
y = torch_inverse(x)
z = torch_matmul(x, y)
torch_max(torch_abs(z - torch_eye(4)$expand_as(x))) # Max non-zero

}
}
}
