% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_finfo}
\alias{torch_finfo}
\title{Float dtype info (range, precision)}
\usage{
torch_finfo(dtype = torch_float32)
}
\arguments{
\item{dtype}{A torch float dtype.}
}
\value{
Named list with bits, eps, max, min, tiny.
}
\description{
Float dtype info (range, precision)
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
