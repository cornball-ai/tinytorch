% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linspace}
\alias{torch_linspace}
\title{Create a tensor with evenly spaced values between start and end}
\usage{
torch_linspace(start, end, steps = 100L, dtype = NULL, device = NULL)
}
\arguments{
\item{start}{Start value.}

\item{end}{End value.}

\item{steps}{Number of points. Default 100.}

\item{dtype}{Optional torch dtype.}

\item{device}{Ignored (CPU only).}
}
\value{
A torch_tensor.
}
\description{
Create a tensor with evenly spaced values between start and end
}
