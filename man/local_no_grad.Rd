% tinyrox says don't edit this manually, but it can't stop you!
\name{local_no_grad}
\alias{local_no_grad}
\title{Local no grad}
\usage{
local_no_grad(.local_envir = parent.frame())
}
\arguments{
\item{.local_envir}{Parameter passed to the underlying ATen operator.}
}
\value{
No return value, called for side effects.
}
\description{
Local no grad
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
