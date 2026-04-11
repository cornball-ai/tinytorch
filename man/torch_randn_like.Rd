% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_randn_like}
\alias{torch_randn_like}
\title{Create a tensor of random normal values with same shape/dtype as input}
\usage{
torch_randn_like(input, dtype = NULL)
}
\arguments{
\item{input}{A `torch_tensor`.}

\item{dtype}{Optional dtype override.}
}
\value{
A torch_tensor.
}
\description{
Create a tensor of random normal values with same shape/dtype as input
}
\examples{
\donttest{
if (torch_is_installed()) {
x <- torch_randn(c(2, 3))
torch_randn_like(x)
}
}
}
