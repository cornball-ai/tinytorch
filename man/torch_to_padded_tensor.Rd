% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_to_padded_tensor}
\alias{torch_to_padded_tensor}
\title{To padded tensor}
\usage{
torch_to_padded_tensor(self, padding, output_size = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{padding}{A numeric value.}

\item{output_size}{An integer vector. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::to_padded_tensor}.
}
