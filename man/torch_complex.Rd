% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_complex}
\alias{torch_complex}
\title{Create a complex tensor from real and imaginary parts}
\usage{
torch_complex(real, imag)
}
\arguments{
\item{real}{Real part tensor.}

\item{imag}{Imaginary part tensor.}
}
\value{
A torch_tensor.
}
\description{
Create a complex tensor from real and imaginary parts
}
\examples{
\donttest{
if (torch_is_installed()) {

real <- torch_tensor(c(1, 2), dtype=torch_float32())
imag <- torch_tensor(c(3, 4), dtype=torch_float32())
z <- torch_complex(real, imag)
z
z$dtype
}
}
}
