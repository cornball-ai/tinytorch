% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_lstm_cell}
\alias{torch_lstm_cell}
\title{Lstm cell}
\usage{
torch_lstm_cell(input, hx, w_ih, w_hh, b_ih = NULL, b_hh = NULL)
}
\arguments{
\item{input}{A torch_tensor.}

\item{hx}{A list of torch_tensors.}

\item{w_ih}{A torch_tensor.}

\item{w_hh}{A torch_tensor.}

\item{b_ih}{A torch_tensor. Optional.}

\item{b_hh}{A torch_tensor. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::lstm_cell}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
