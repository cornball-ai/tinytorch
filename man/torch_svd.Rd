% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_svd}
\alias{torch_svd}
\title{Svd}
\usage{
torch_svd(self, some = TRUE, compute_uv = TRUE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{some}{A logical value.}

\item{compute_uv}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::svd}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(5, 3))
a
out = torch_svd(a)
u = out[[1]]
s = out[[2]]
v = out[[3]]
torch_dist(a, torch_mm(torch_mm(u, torch_diag(s)), v$t()))
a_big = torch_randn(c(7, 5, 3))
out = torch_svd(a_big)
u = out[[1]]
s = out[[2]]
v = out[[3]]
torch_dist(a_big, torch_matmul(torch_matmul(u, torch_diag_embed(s)), v$transpose(-2, -1)))
}
}
}
