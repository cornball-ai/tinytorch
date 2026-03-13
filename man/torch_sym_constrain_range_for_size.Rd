% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_sym_constrain_range_for_size}
\alias{torch_sym_constrain_range_for_size}
\title{Sym constrain range for size}
\usage{
torch_sym_constrain_range_for_size(size, min = NULL, max = NULL)
}
\arguments{
\item{size}{A numeric scalar.}

\item{min}{An integer. Optional.}

\item{max}{An integer. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::sym_constrain_range_for_size}.
}
