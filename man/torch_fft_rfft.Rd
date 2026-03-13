% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_fft_rfft}
\alias{torch_fft_rfft}
\title{Fft rfft}
\usage{
torch_fft_rfft(self, n = NULL, dim = -1, norm = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{n}{An integer. Optional.}

\item{dim}{An integer.}

\item{norm}{A character string. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::fft_rfft}.
}
