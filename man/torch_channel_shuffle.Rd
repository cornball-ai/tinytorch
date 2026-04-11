% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_channel_shuffle}
\alias{torch_channel_shuffle}
\title{Channel shuffle}
\usage{
torch_channel_shuffle(self, groups)
}
\arguments{
\item{self}{A torch_tensor.}

\item{groups}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::channel_shuffle}.
}
\examples{
\donttest{
if (torch_is_installed()) {

input <- torch_randn(c(1, 4, 2, 2))
print(input)
output <- torch_channel_shuffle(input, 2)
print(output)
}
}
}
