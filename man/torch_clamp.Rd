% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_clamp}
\alias{torch_clamp}
\title{Clamp tensor values to a range}
\usage{
torch_clamp(self, min = NULL, max = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{min}{Minimum value (optional).}

\item{max}{Maximum value (optional).}
}
\value{
A torch_tensor.
}
\description{
Clamp tensor values to a range
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_clamp(a, min=-0.5, max=0.5)


a = torch_randn(c(4))
a
torch_clamp(a, min=0.5)


a = torch_randn(c(4))
a
torch_clamp(a, max=0.5)
}
}
}
