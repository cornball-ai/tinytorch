% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_lerp}
\alias{torch_lerp}
\title{Lerp}
\usage{
torch_lerp(self, end, weight)
}
\arguments{
\item{self}{A torch_tensor.}

\item{end}{A torch_tensor.}

\item{weight}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::lerp}.
}
