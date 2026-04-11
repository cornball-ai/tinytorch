% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_dot}
\alias{torch_dot}
\title{Dot}
\usage{
torch_dot(self, tensor)
}
\arguments{
\item{self}{A torch_tensor.}

\item{tensor}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::dot}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_dot(torch_tensor(c(2, 3)), torch_tensor(c(2, 1)))
}
}
}
