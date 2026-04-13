% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_dropout}
\alias{nnf_dropout}
\title{Nnf dropout}
\usage{
nnf_dropout(input, p, train)
}
\arguments{
\item{input}{A `torch_tensor`.}

\item{p}{Parameter passed to the underlying ATen operator.}

\item{train}{Parameter passed to the underlying ATen operator.}
}
\value{
A `torch_tensor`.
}
\description{
Nnf dropout
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
