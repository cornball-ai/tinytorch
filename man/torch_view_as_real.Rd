% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_view_as_real}
\alias{torch_view_as_real}
\title{View a complex tensor as a real tensor with extra dimension}
\usage{
torch_view_as_real(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
View a complex tensor as a real tensor with extra dimension
}
\examples{
\donttest{
if (torch_is_installed()) {

if (FALSE) {
x <- torch_randn(4, dtype=torch_cfloat())
x
torch_view_as_real(x)
}
}
}
}
