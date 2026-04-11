% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_baddbmm}
\alias{torch_baddbmm}
\title{Baddbmm}
\usage{
torch_baddbmm(self, batch1, batch2, beta = 1, alpha = 1, out_dtype = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{batch1}{A torch_tensor.}

\item{batch2}{A torch_tensor.}

\item{beta}{A numeric scalar.}

\item{alpha}{A numeric scalar.}

\item{out_dtype}{Optional output dtype for mixed-precision (PyTorch 2.8+).}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::baddbmm}.
}
\examples{
\donttest{
if (torch_is_installed()) {

M = torch_randn(c(10, 3, 5))
batch1 = torch_randn(c(10, 3, 4))
batch2 = torch_randn(c(10, 4, 5))
torch_baddbmm(M, batch1, batch2)
}
}
}
