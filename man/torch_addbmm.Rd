% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_addbmm}
\alias{torch_addbmm}
\title{Addbmm}
\usage{
torch_addbmm(self, batch1, batch2, beta = 1, alpha = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{batch1}{A torch_tensor.}

\item{batch2}{A torch_tensor.}

\item{beta}{A numeric scalar.}

\item{alpha}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::addbmm}.
}
\examples{
\donttest{
if (torch_is_installed()) {

M = torch_randn(c(3, 5))
batch1 = torch_randn(c(10, 3, 4))
batch2 = torch_randn(c(10, 4, 5))
torch_addbmm(M, batch1, batch2)
}
}
}
