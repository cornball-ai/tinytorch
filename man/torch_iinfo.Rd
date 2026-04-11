% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_iinfo}
\alias{torch_iinfo}
\title{Integer dtype info (range)}
\usage{
torch_iinfo(dtype = torch_int32)
}
\arguments{
\item{dtype}{A torch integer dtype.}
}
\value{
Named list with bits, max, min.
}
\description{
Integer dtype info (range)
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
