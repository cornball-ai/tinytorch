% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_tensor}
\alias{torch_tensor}
\title{Create a tensor from R data}
\usage{
torch_tensor(data, dtype = NULL, device = NULL, requires_grad = FALSE)
}
\arguments{
\item{data}{An R numeric, integer, or logical vector/matrix/array.}

\item{dtype}{Optional torch dtype (e.g. torch_float).}

\item{device}{Ignored for now (CPU only).}

\item{requires_grad}{Logical. Track gradients for this tensor. Default FALSE.}
}
\value{
A torch_tensor object.
}
\description{
Create a tensor from R data
}
\examples{
\dontrun{
torch_tensor(c(1, 2, 3))
torch_tensor(matrix(1:6, 2, 3))
}
}
