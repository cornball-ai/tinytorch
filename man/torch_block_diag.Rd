% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_block_diag}
\alias{torch_block_diag}
\title{Block diag}
\usage{
torch_block_diag(tensors)
}
\arguments{
\item{tensors}{A list of torch_tensors.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::block_diag}.
}
\examples{
\donttest{
if (torch_is_installed()) {

A <- torch_tensor(rbind(c(0, 1), c(1, 0)))
B <- torch_tensor(rbind(c(3, 4, 5), c(6, 7, 8)))
C <- torch_tensor(7)
D <- torch_tensor(c(1, 2, 3))
E <- torch_tensor(rbind(4, 5, 6))
torch_block_diag(list(A, B, C, D, E))
}
}
}
