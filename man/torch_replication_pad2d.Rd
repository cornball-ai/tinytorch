% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_replication_pad2d}
\alias{torch_replication_pad2d}
\title{Replication pad2d}
\usage{
torch_replication_pad2d(self, padding)
}
\arguments{
\item{self}{A torch_tensor.}

\item{padding}{(int[4]).}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::replication_pad2d}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
