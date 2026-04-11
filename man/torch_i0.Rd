% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_i0}
\alias{torch_i0}
\title{I0}
\usage{
torch_i0(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::i0}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_i0(torch_arange(start = 0, end = 5, dtype=torch_float32()))
}
}
}
