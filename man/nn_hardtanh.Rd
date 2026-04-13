% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_hardtanh}
\alias{nn_hardtanh}
\title{Nn hardtanh}
\usage{
nn_hardtanh(min_val = -1, max_val = 1)
}
\arguments{
\item{min_val}{Parameter passed to the underlying ATen operator.}

\item{max_val}{Parameter passed to the underlying ATen operator.}
}
\value{
An `nn_module` instance.
}
\description{
Nn hardtanh
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
