% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_logical_xor}
\alias{torch_logical_xor}
\title{Logical xor}
\usage{
torch_logical_xor(self, other)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::logical_xor}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_logical_xor(torch_tensor(c(TRUE, FALSE, TRUE)), torch_tensor(c(TRUE, FALSE, FALSE)))
a = torch_tensor(c(0, 1, 10, 0), dtype=torch_int8())
b = torch_tensor(c(4, 0, 1, 0), dtype=torch_int8())
torch_logical_xor(a, b)
torch_logical_xor(a$to(dtype=torch_double()), b$to(dtype=torch_double()))
torch_logical_xor(a$to(dtype=torch_double()), b)
}
}
}
