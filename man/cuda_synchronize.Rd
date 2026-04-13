% tinyrox says don't edit this manually, but it can't stop you!
\name{cuda_synchronize}
\alias{cuda_synchronize}
\title{Cuda synchronize}
\usage{
cuda_synchronize(device = NULL)
}
\arguments{
\item{device}{Parameter passed to the underlying ATen operator.}
}
\value{
No return value, called for side effects.
}
\description{
Cuda synchronize
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
