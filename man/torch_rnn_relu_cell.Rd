% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_rnn_relu_cell}
\alias{torch_rnn_relu_cell}
\title{Rnn relu cell}
\usage{
torch_rnn_relu_cell(input, hx, w_ih, w_hh, b_ih = NULL, b_hh = NULL)
}
\arguments{
\item{input}{A torch_tensor.}

\item{hx}{A torch_tensor.}

\item{w_ih}{A torch_tensor.}

\item{w_hh}{A torch_tensor.}

\item{b_ih}{A torch_tensor. Optional.}

\item{b_hh}{A torch_tensor. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::rnn_relu_cell}.
}
