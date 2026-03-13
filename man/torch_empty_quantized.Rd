% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_empty_quantized}
\alias{torch_empty_quantized}
\title{Empty quantized}
\usage{
torch_empty_quantized(size, qtensor, dtype = NULL, device = NULL,
                      memory_format = NULL)
}
\arguments{
\item{size}{An integer vector.}

\item{qtensor}{A torch_tensor.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{device}{A device string (e.g., "cpu"). Optional.}

\item{memory_format}{A memory format. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::empty_quantized}.
}
