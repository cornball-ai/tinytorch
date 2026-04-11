% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_repeat_interleave}
\alias{torch_repeat_interleave}
\title{torch_repeat_interleave}
\usage{
torch_repeat_interleave(self, repeats, dim = NULL)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{repeats}{Integer or `torch_tensor` of repeat counts.}

\item{dim}{Integer dimension.}
}
\description{
torch_repeat_interleave
}
\examples{
\donttest{
if (torch_is_installed()) {

x = torch_tensor(c(1, 2, 3))
x$repeat_interleave(2)
y = torch_tensor(matrix(c(1, 2, 3, 4), ncol = 2, byrow=TRUE))
torch_repeat_interleave(y, 2)
torch_repeat_interleave(y, 3, dim=1)
torch_repeat_interleave(y, torch_tensor(c(1, 2)), dim=1)

}
}
}
