% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_stft}
\alias{torch_stft}
\title{Short-time Fourier Transform}
\usage{
torch_stft(input, n_fft, hop_length = NULL, win_length = NULL, window = NULL,
           center = TRUE, normalized = FALSE, onesided = NULL,
           return_complex = TRUE)
}
\arguments{
\item{input}{Input tensor.}

\item{n_fft}{FFT size.}

\item{hop_length}{Hop length. Default n_fft / 4.}

\item{win_length}{Window length. Default n_fft.}

\item{window}{Optional window tensor.}

\item{center}{Whether to pad input on both sides. Default TRUE.}

\item{normalized}{Whether to normalize. Default FALSE.}

\item{onesided}{Whether to return one-sided. Default TRUE.}

\item{return_complex}{Whether to return complex output. Default TRUE.}
}
\value{
A torch_tensor.
}
\description{
Short-time Fourier Transform
}
\examples{
\donttest{
if (torch_is_installed()) {
x <- torch_randn(512)
torch_stft(x, n_fft = 64L, hop_length = 16L,
           window = torch_hann_window(64L), return_complex = TRUE)
}
}
}
