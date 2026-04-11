% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_qr}
\alias{torch_qr}
\title{Qr}
\usage{
torch_qr(self, some = TRUE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{some}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::qr}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_tensor(matrix(c(12., -51, 4, 6, 167, -68, -4, 24, -41), ncol = 3, byrow = TRUE))
out = torch_qr(a)
q = out[[1]]
r = out[[2]]
torch_mm(q, r)$round()
torch_mm(q$t(), q)$round()
}
}
}
