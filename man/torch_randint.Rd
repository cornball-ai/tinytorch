% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_randint}
\alias{torch_randint}
\title{Randint}
\usage{
torch_randint(high, size, dtype = torch_long, device = NULL)
}
\arguments{
\item{high}{An integer.}

\item{size}{An integer vector.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{device}{A device string (e.g., "cpu"). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::randint}.
}
