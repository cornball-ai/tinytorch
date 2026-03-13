% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_hamming_window}
\alias{torch_hamming_window}
\title{Hamming window}
\usage{
torch_hamming_window(window_length, dtype = NULL, device = NULL)
}
\arguments{
\item{window_length}{An integer.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{device}{A device string (e.g., "cpu"). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::hamming_window}.
}
