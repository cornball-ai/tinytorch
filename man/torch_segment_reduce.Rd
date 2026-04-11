% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_segment_reduce}
\alias{torch_segment_reduce}
\title{Segment reduce}
\usage{
torch_segment_reduce(data, reduce, lengths = NULL, indices = NULL,
                     offsets = NULL, axis = 0, unsafe = FALSE, initial = NULL)
}
\arguments{
\item{data}{A torch_tensor.}

\item{reduce}{A character string.}

\item{lengths}{A torch_tensor. Optional.}

\item{indices}{A torch_tensor. Optional.}

\item{offsets}{A torch_tensor. Optional.}

\item{axis}{An integer.}

\item{unsafe}{A logical value.}

\item{initial}{A numeric scalar. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::segment_reduce}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
