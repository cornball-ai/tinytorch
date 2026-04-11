% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_masked_select}
\alias{torch_masked_select}
\title{Masked select}
\usage{
torch_masked_select(self, mask)
}
\arguments{
\item{self}{A torch_tensor.}

\item{mask}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::masked_select}.
}
\examples{
\donttest{
if (torch_is_installed()) {

x = torch_randn(c(3, 4))
x
mask = x$ge(0.5)
mask
torch_masked_select(x, mask)
}
}
}
