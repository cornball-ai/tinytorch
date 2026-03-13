% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_fft_ifftn}
\alias{torch_fft_ifftn}
\title{Fft ifftn}
\usage{
torch_fft_ifftn(self, s = NULL, dim = NULL, norm = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{s}{(int[1]). Optional.}

\item{dim}{(int[1]). Optional.}

\item{norm}{A character string. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::fft_ifftn}.
}
