% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_alpha_dropout}
\alias{nnf_alpha_dropout}
\title{Nnf alpha dropout}
\usage{
nnf_alpha_dropout(input, p, train)
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
Nnf alpha dropout
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
