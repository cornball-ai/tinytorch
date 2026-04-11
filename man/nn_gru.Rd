% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_gru}
\alias{nn_gru}
\title{GRU module}
\usage{
nn_gru(input_size, hidden_size, num_layers = 1L, bias = TRUE,
       batch_first = FALSE, dropout = 0, bidirectional = FALSE)
}
\arguments{
\item{input_size}{Input feature size.}

\item{hidden_size}{Hidden state size.}

\item{num_layers}{Number of layers. Default 1.}

\item{bias}{Use bias. Default TRUE.}

\item{batch_first}{Batch-first input. Default FALSE.}

\item{dropout}{Dropout rate. Default 0.}

\item{bidirectional}{Bidirectional. Default FALSE.}
}
\value{
An nn_module.
}
\description{
GRU module
}
