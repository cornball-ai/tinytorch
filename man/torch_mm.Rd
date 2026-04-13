% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_mm}
\alias{torch_mm}
\title{Matrix-matrix multiplication (2D only)}
\usage{
torch_mm(self, other, out_dtype = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}

\item{out_dtype}{Optional output dtype for mixed-precision (PyTorch 2.8+).}
}
\value{
A `torch_tensor`.
}
\description{
Matrix-matrix multiplication (2D only)
}
\examples{
\donttest{
if (torch_is_installed()) {
a <- torch_randn(c(2, 3))
b <- torch_randn(c(3, 4))
torch_mm(a, b)
}
}
}
