% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_conv_tbc}
\alias{torch_conv_tbc}
\title{Conv tbc}
\usage{
torch_conv_tbc(self, weight, bias, pad = 0)
}
\arguments{
\item{self}{A torch_tensor.}

\item{weight}{A torch_tensor.}

\item{bias}{A torch_tensor.}

\item{pad}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::conv_tbc}.
}
