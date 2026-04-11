% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_zeros_like}
\alias{torch_zeros_like}
\title{Create a tensor of zeros with same shape/dtype as input}
\usage{
torch_zeros_like(input, dtype = NULL)
}
\arguments{
\item{input}{A `torch_tensor`.}

\item{dtype}{Optional dtype override.}
}
\value{
A torch_tensor.
}
\description{
Create a tensor of zeros with same shape/dtype as input
}
\examples{
\donttest{
if (torch_is_installed()) {

input = torch_empty(c(2, 3))
torch_zeros_like(input)
}
}
}
