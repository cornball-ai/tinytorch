% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_gru}
\alias{torch_gru}
\title{Gru}
\usage{
torch_gru(input, hx, params, has_biases, num_layers, dropout, train,
          bidirectional, batch_first)
}
\arguments{
\item{input}{A torch_tensor.}

\item{hx}{A torch_tensor.}

\item{params}{A list of torch_tensors.}

\item{has_biases}{A logical value.}

\item{num_layers}{An integer.}

\item{dropout}{A numeric value.}

\item{train}{A logical value.}

\item{bidirectional}{A logical value.}

\item{batch_first}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::gru}.
}
