% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_pad_sequence}
\alias{torch_pad_sequence}
\title{Pad sequence}
\usage{
torch_pad_sequence(sequences, batch_first = FALSE, padding_value = 0,
                   padding_side = "right")
}
\arguments{
\item{sequences}{A list of torch_tensors.}

\item{batch_first}{A logical value.}

\item{padding_value}{A numeric value.}

\item{padding_side}{A character string.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::pad_sequence}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
