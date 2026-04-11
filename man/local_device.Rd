% tinyrox says don't edit this manually, but it can't stop you!
\name{local_device}
\alias{local_device}
\title{Local device}
\usage{
local_device(device, .local_envir = parent.frame())
}
\arguments{
\item{device}{Parameter passed to the underlying ATen operator.}

\item{.local_envir}{Parameter passed to the underlying ATen operator.}
}
\description{
Local device
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
