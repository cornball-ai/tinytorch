% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_istft}
\alias{torch_istft}
\title{Inverse Short-time Fourier Transform}
\usage{
torch_istft(input, n_fft, hop_length = NULL, win_length = NULL, window = NULL,
            center = TRUE, normalized = FALSE, onesided = TRUE, length = NULL,
            return_complex = FALSE)
}
\arguments{
\item{input}{Input tensor.}

\item{n_fft}{FFT size.}

\item{hop_length}{Hop length. Default n_fft / 4.}

\item{win_length}{Window length. Default n_fft.}

\item{window}{Optional window tensor.}

\item{center}{Whether input was centered. Default TRUE.}

\item{normalized}{Whether input was normalized. Default FALSE.}

\item{onesided}{Whether input is one-sided. Default TRUE.}

\item{length}{Expected output length. Default NULL.}

\item{return_complex}{Whether to return complex output. Default FALSE.}
}
\value{
A torch_tensor.
}
\description{
Inverse Short-time Fourier Transform
}
