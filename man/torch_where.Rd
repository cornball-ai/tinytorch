% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_where}
\alias{torch_where}
\title{Element-wise selection based on condition}
\usage{
torch_where(condition, self, other)
}
\arguments{
\item{condition}{Boolean tensor.}

\item{self}{Tensor for TRUE elements.}

\item{other}{Tensor for FALSE elements.}
}
\value{
A torch_tensor.
}
\description{
Element-wise selection based on condition
}
\examples{
\donttest{
if (torch_is_installed()) {


x = torch_randn(c(3, 2))
y = torch_ones(c(3, 2))
x
torch_where(x > 0, x, y)




}
}
}
