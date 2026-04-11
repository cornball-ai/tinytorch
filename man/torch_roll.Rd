% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_roll}
\alias{torch_roll}
\title{Roll}
\usage{
torch_roll(self, shifts, dims = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{shifts}{(int[1]).}

\item{dims}{(int[1]).}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::roll}.
}
\examples{
\donttest{
if (torch_is_installed()) {

x = torch_tensor(c(1, 2, 3, 4, 5, 6, 7, 8))$view(c(4, 2))
x
torch_roll(x, 1, 1)
torch_roll(x, -1, 1)
torch_roll(x, shifts=list(2, 1), dims=list(1, 2))
}
}
}
