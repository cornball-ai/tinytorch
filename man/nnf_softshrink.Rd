% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_softshrink}
\alias{nnf_softshrink}
\title{Nnf softshrink}
\usage{
nnf_softshrink(self, lambd)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{lambd}{Parameter passed to the underlying ATen operator.}
}
\value{
A `torch_tensor`.
}
\description{
Nnf softshrink
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
