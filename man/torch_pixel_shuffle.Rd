% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_pixel_shuffle}
\alias{torch_pixel_shuffle}
\title{Pixel shuffle}
\usage{
torch_pixel_shuffle(self, upscale_factor)
}
\arguments{
\item{self}{A torch_tensor.}

\item{upscale_factor}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::pixel_shuffle}.
}
\examples{
\donttest{
if (torch_is_installed()) {

input = torch_randn(c(1, 9, 4, 4))
output = nnf_pixel_shuffle(input, 3)
print(output$size())
}
}
}
