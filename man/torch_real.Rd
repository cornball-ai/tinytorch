% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_real}
\alias{torch_real}
\title{Extract real part of a complex tensor}
\usage{
torch_real(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Extract real part of a complex tensor
}
\examples{
\donttest{
if (torch_is_installed()) {
  # Requires complex tensor support; see PyTorch docs
}
}
}
