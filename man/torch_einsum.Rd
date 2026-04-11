% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_einsum}
\alias{torch_einsum}
\title{Einsum}
\usage{
torch_einsum(equation, tensors, path = NULL)
}
\arguments{
\item{equation}{A character string.}

\item{tensors}{A list of torch_tensors.}

\item{path}{An integer vector. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::einsum}.
}
\examples{
\donttest{
if (torch_is_installed()) {

x = torch_randn(c(5))
y = torch_randn(c(4))
torch_einsum('i,j->ij', list(x, y))  # outer product
A = torch_randn(c(3,5,4))
l = torch_randn(c(2,5))
r = torch_randn(c(2,4))
torch_einsum('bn,anm,bm->ba', list(l, A, r)) # compare torch_nn$functional$bilinear
As = torch_randn(c(3,2,5))
Bs = torch_randn(c(3,5,4))
torch_einsum('bij,bjk->bik', list(As, Bs)) # batch matrix multiplication
A = torch_randn(c(3, 3))
torch_einsum('ii->i', list(A)) # diagonal
A = torch_randn(c(4, 3, 3))
torch_einsum('...ii->...i', list(A)) # batch diagonal
A = torch_randn(c(2, 3, 4, 5))
torch_einsum('...ij->...ji', list(A))$shape # batch permute

}
}
}
