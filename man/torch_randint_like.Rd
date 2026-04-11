% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_randint_like}
\alias{torch_randint_like}
\title{Randint like}
\usage{
torch_randint_like(input, high, dtype = NULL, device = NULL,
                   memory_format = NULL)
}
\arguments{
\item{high}{An integer.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{device}{A device string (e.g., "cpu"). Optional.}

\item{memory_format}{A memory format. Optional.}

\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::randint_like}.
}
