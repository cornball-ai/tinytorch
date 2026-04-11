% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_clip_}
\alias{torch_clip_}
\title{Clip}
\usage{
torch_clip_(self, min = NULL, max = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{min}{A numeric scalar. Optional.}

\item{max}{A numeric scalar. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::clip_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
