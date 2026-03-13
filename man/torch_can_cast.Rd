% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_can_cast}
\alias{torch_can_cast}
\title{Can cast}
\usage{
torch_can_cast(from_, to)
}
\arguments{
\item{from_}{A torch dtype (e.g., torch_float32).}

\item{to}{A torch dtype (e.g., torch_float32).}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::can_cast}.
}
