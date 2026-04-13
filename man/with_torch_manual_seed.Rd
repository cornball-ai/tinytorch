% tinyrox says don't edit this manually, but it can't stop you!
\name{with_torch_manual_seed}
\alias{with_torch_manual_seed}
\title{With torch manual seed}
\usage{
with_torch_manual_seed(seed, code)
}
\arguments{
\item{seed}{Parameter passed to the underlying ATen operator.}

\item{code}{Parameter passed to the underlying ATen operator.}
}
\value{
The result of evaluating `code`.
}
\description{
With torch manual seed
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
