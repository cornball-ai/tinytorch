% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_imag}
\alias{torch_imag}
\title{Extract imaginary part of a complex tensor}
\usage{
torch_imag(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Extract imaginary part of a complex tensor
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_imag(torch_tensor(c(-1 + 1i, -2 + 2i, 3 - 3i)))

}
}
}
