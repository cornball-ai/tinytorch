% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_ger}
\alias{torch_ger}
\title{Ger}
\usage{
torch_ger(self, vec2)
}
\arguments{
\item{self}{A torch_tensor.}

\item{vec2}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::ger}.
}
\examples{
\donttest{
if (torch_is_installed()) {

v1 = torch_arange(1., 5.)
v2 = torch_arange(1., 4.)
torch_ger(v1, v2)
}
}
}
