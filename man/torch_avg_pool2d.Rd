% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_avg_pool2d}
\alias{torch_avg_pool2d}
\title{Avg pool2d}
\usage{
torch_avg_pool2d(self, kernel_size, stride = NULL, padding = 0,
                 ceil_mode = FALSE, count_include_pad = TRUE,
                 divisor_override = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{kernel_size}{(int[2]).}

\item{stride}{(int[2]).}

\item{padding}{(int[2]).}

\item{ceil_mode}{A logical value.}

\item{count_include_pad}{A logical value.}

\item{divisor_override}{An integer. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::avg_pool2d}.
}
