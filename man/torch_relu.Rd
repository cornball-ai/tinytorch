% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_relu}
\alias{torch_relu}
\title{ReLU activation (namespace-level)}
\usage{
torch_relu(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\description{
ReLU activation (namespace-level)
}
\examples{
\donttest{
if (torch_is_installed()) {
torch_relu(torch_tensor(c(-1, 0, 1)))
}
}
}
