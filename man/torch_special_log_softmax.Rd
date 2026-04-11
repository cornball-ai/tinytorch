% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_special_log_softmax}
\alias{torch_special_log_softmax}
\title{Special log softmax}
\usage{
torch_special_log_softmax(self, dim, dtype = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{An integer.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::special_log_softmax}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
