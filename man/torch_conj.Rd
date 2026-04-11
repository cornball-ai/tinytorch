% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_conj}
\alias{torch_conj}
\title{torch_conj}
\usage{
torch_conj(self)
}
\arguments{
\item{self}{A `torch_tensor`.}
}
\description{
torch_conj
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_conj(torch_tensor(c(-1 + 1i, -2 + 2i, 3 - 3i)))

}
}
}
