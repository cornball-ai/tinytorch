% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_leaky_relu}
\alias{nnf_leaky_relu}
\title{Leaky ReLU activation}
\usage{
nnf_leaky_relu(self, negative_slope = 0.01)
}
\arguments{
\item{self}{A torch_tensor.}

\item{negative_slope}{Negative slope coefficient. Default 0.01.}
}
\description{
Leaky ReLU activation
}
\examples{
\donttest{
nnf_leaky_relu(torch_tensor(c(-1, 0, 1)))
}
}
