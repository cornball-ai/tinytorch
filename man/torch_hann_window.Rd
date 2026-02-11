% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_hann_window}
\alias{torch_hann_window}
\title{Create a Hann window tensor}
\usage{
torch_hann_window(window_length, periodic = TRUE, dtype = NULL, device = NULL)
}
\arguments{
\item{window_length}{Length of the window.}

\item{periodic}{Whether the window is periodic. Default TRUE.}

\item{dtype}{Optional torch dtype.}

\item{device}{Ignored (CPU only).}
}
\value{
A torch_tensor.
}
\description{
Create a Hann window tensor
}
