% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_polar}
\alias{torch_polar}
\title{Create a complex tensor from polar coordinates}
\usage{
torch_polar(abs, angle)
}
\arguments{
\item{abs}{Absolute value tensor.}

\item{angle}{Angle tensor.}
}
\value{
A torch_tensor.
}
\description{
Create a complex tensor from polar coordinates
}
\examples{
\donttest{
if (torch_is_installed()) {

abs <- torch_tensor(c(1, 2), dtype=torch_float64())
angle <- torch_tensor(c(pi / 2, 5 * pi / 4), dtype=torch_float64())
z <- torch_polar(abs, angle)
z
}
}
}
