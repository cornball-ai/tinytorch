% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_heaviside}
\alias{torch_heaviside}
\title{Heaviside}
\usage{
torch_heaviside(self, values)
}
\arguments{
\item{self}{A torch_tensor.}

\item{values}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::heaviside}.
}
\examples{
\donttest{
if (torch_is_installed()) {

input <- torch_tensor(c(-1.5, 0, 2.0))
values <- torch_tensor(c(0.5))
torch_heaviside(input, values)
values <- torch_tensor(c(1.2, -2.0, 3.5))
torch_heaviside(input, values)
}
}
}
