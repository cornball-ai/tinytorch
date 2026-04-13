% tinyrox says don't edit this manually, but it can't stop you!
\name{local_enable_grad}
\alias{local_enable_grad}
\title{Local enable grad}
\usage{
local_enable_grad(.local_envir = parent.frame())
}
\arguments{
\item{.local_envir}{Parameter passed to the underlying ATen operator.}
}
\value{
No return value, called for side effects.
}
\description{
Local enable grad
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
