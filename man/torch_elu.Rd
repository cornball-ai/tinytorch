% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_elu}
\alias{torch_elu}
\title{Elu}
\usage{
torch_elu(self, alpha = 1, scale = 1, input_scale = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{alpha}{A numeric scalar.}

\item{scale}{A numeric scalar.}

\item{input_scale}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::elu}.
}
