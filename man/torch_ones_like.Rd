% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_ones_like}
\alias{torch_ones_like}
\title{Create a tensor of ones with same shape/dtype as input}
\usage{
torch_ones_like(self, dtype = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dtype}{Optional dtype override.}
}
\value{
A torch_tensor.
}
\description{
Create a tensor of ones with same shape/dtype as input
}
\examples{
\donttest{
if (torch_is_installed()) {

input = torch_empty(c(2, 3))
torch_ones_like(input)
}
}
}
