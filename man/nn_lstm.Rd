% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_lstm}
\alias{nn_lstm}
\title{LSTM module}
\usage{
nn_lstm(
  input_size,
  hidden_size,
  num_layers = 1L,
  batch_first = TRUE,
  dropout = 0,
  bidirectional = FALSE,
  bias = TRUE
)
}
\arguments{
\item{input_size}{Integer.}

\item{hidden_size}{Integer.}

\item{num_layers}{Integer. Default 1.}

\item{batch_first}{Logical. Default TRUE.}

\item{dropout}{Double. Default 0.}

\item{bidirectional}{Logical. Default FALSE.}

\item{bias}{Logical. Default TRUE.}
}
\value{
An nn_module instance.
}
\description{
LSTM module
}
