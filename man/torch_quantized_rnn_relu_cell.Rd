% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_quantized_rnn_relu_cell}
\alias{torch_quantized_rnn_relu_cell}
\title{Quantized rnn relu cell}
\usage{
torch_quantized_rnn_relu_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih,
                              packed_hh, col_offsets_ih, col_offsets_hh,
                              scale_ih, scale_hh, zero_point_ih, zero_point_hh)
}
\arguments{
\item{input}{A torch_tensor.}

\item{hx}{A torch_tensor.}

\item{w_ih}{A torch_tensor.}

\item{w_hh}{A torch_tensor.}

\item{b_ih}{A torch_tensor.}

\item{b_hh}{A torch_tensor.}

\item{packed_ih}{A torch_tensor.}

\item{packed_hh}{A torch_tensor.}

\item{col_offsets_ih}{A torch_tensor.}

\item{col_offsets_hh}{A torch_tensor.}

\item{scale_ih}{A numeric scalar.}

\item{scale_hh}{A numeric scalar.}

\item{zero_point_ih}{A numeric scalar.}

\item{zero_point_hh}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::quantized_rnn_relu_cell}.
}
