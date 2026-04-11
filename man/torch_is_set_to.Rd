% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_is_set_to}
\alias{torch_is_set_to}
\title{Is set to}
\usage{
torch_is_set_to(self, tensor)
}
\arguments{
\item{self}{A torch_tensor.}

\item{tensor}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::is_set_to}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
