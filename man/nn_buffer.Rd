% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_buffer}
\alias{nn_buffer}
\title{Tag a tensor as a buffer (not a parameter)}
\usage{
nn_buffer(data, persistent = TRUE)
}
\arguments{
\item{data}{A torch_tensor.}

\item{persistent}{Ignored.}
}
\value{
The tensor with additional class tag.
}
\description{
Tag a tensor as a buffer (not a parameter)
}
\examples{
\donttest{
if (torch_is_installed()) {
b <- nn_buffer(torch_zeros(3))
b
}
}
}
