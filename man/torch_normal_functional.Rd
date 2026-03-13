% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_normal_functional}
\alias{torch_normal_functional}
\title{Normal functional}
\usage{
torch_normal_functional(self, mean = 0, std = 1, generator = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{mean}{A numeric value.}

\item{std}{A numeric value.}

\item{generator}{A random number generator. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::normal_functional}.
}
