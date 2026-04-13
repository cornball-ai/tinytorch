% tinyrox says don't edit this manually, but it can't stop you!
\name{local_torch_manual_seed}
\alias{local_torch_manual_seed}
\title{Local torch manual seed}
\usage{
local_torch_manual_seed(seed, .local_envir = parent.frame())
}
\arguments{
\item{seed}{Parameter passed to the underlying ATen operator.}

\item{.local_envir}{Parameter passed to the underlying ATen operator.}
}
\value{
No return value, called for side effects.
}
\description{
Local torch manual seed
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
