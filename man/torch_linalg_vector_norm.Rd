% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_vector_norm}
\alias{torch_linalg_vector_norm}
\alias{linalg_vector_norm}
\title{Linalg vector norm}
\usage{
torch_linalg_vector_norm(self, ord = 2, dim = NULL, keepdim = FALSE,
                         dtype = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{ord}{A numeric scalar.}

\item{dim}{(int[1]). Optional.}

\item{keepdim}{A logical value.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_vector_norm}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
