% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_init_orthogonal_}
\alias{nn_init_orthogonal_}
\title{Orthogonal initialization}
\usage{
nn_init_orthogonal_(tensor, gain = 1)
}
\arguments{
\item{tensor}{A 2D+ tensor.}

\item{gain}{Scaling factor. Default 1.}
}
\value{
tensor (modified in place).
}
\description{
Orthogonal initialization
}
\examples{
\donttest{
if (torch_is_installed()) {
w <- torch_empty(3, 5)
nn_init_orthogonal_(w)
}
}
}
