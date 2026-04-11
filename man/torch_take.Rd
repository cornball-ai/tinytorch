% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_take}
\alias{torch_take}
\title{Take}
\usage{
torch_take(self, index)
}
\arguments{
\item{self}{A torch_tensor.}

\item{index}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::take}.
}
\examples{
\donttest{
if (torch_is_installed()) {

src = torch_tensor(matrix(c(4,3,5,6,7,8), ncol = 3, byrow = TRUE))
torch_take(src, torch_tensor(c(1, 2, 5), dtype = torch_int64()))
}
}
}
