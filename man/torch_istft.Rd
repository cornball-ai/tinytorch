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
\examples{
\donttest{
if (torch_is_installed()) {
x <- torch_randn(512)
win <- torch_hann_window(64L)
y <- torch_stft(x, n_fft = 64L, hop_length = 16L,
                window = win, return_complex = TRUE)
torch_istft(y, n_fft = 64L, hop_length = 16L, window = win)
}
}
}
