% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_arange}
\alias{torch_arange}
\title{Create a tensor with evenly spaced values}
\usage{
torch_arange(start, end, step = 1L, dtype = NULL, device = NULL)
}
\arguments{
\item{start}{Start value.}

\item{end}{End value (exclusive in PyTorch convention).}

\item{step}{Step between values. Default 1.}

\item{dtype}{Optional torch dtype.}

\item{device}{Ignored (CPU only).}
}
\value{
A torch_tensor.
}
\description{
Create a tensor with evenly spaced values
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_arange(start = 0, end = 5)
torch_arange(1, 4)
torch_arange(1, 2.5, 0.5)
}
}
}
