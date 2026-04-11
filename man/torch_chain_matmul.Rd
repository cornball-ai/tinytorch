% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_chain_matmul}
\alias{torch_chain_matmul}
\title{Chain matmul}
\usage{
torch_chain_matmul(matrices)
}
\arguments{
\item{matrices}{A list of torch_tensors.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::chain_matmul}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(3, 4))
b = torch_randn(c(4, 5))
c = torch_randn(c(5, 6))
d = torch_randn(c(6, 7))
torch_chain_matmul(list(a, b, c, d))
}
}
}
