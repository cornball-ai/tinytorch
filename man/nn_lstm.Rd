% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_lstm}
\alias{nn_lstm}
\title{LSTM module}
\usage{
nn_lstm(input_size, hidden_size, num_layers = 1L, batch_first = TRUE,
        dropout = 0, bidirectional = FALSE, bias = TRUE)
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
\examples{
\donttest{
if (torch_is_installed()) {
rnn <- nn_lstm(10, 20, 2)
input <- torch_randn(5, 3, 10)
h0 <- torch_randn(2, 3, 20)
c0 <- torch_randn(2, 3, 20)
output <- rnn(input, list(h0, c0))
}
}
}
