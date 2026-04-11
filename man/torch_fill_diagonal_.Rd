% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_fill_diagonal_}
\alias{torch_fill_diagonal_}
\title{Fill diagonal}
\usage{
torch_fill_diagonal_(self, fill_value, wrap = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{fill_value}{A numeric scalar.}

\item{wrap}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::fill_diagonal_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
