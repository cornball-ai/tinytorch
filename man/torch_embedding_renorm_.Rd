% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_embedding_renorm_}
\alias{torch_embedding_renorm_}
\title{Embedding renorm}
\usage{
torch_embedding_renorm_(self, indices, max_norm, norm_type)
}
\arguments{
\item{self}{A torch_tensor.}

\item{indices}{A torch_tensor.}

\item{max_norm}{A numeric value.}

\item{norm_type}{A numeric value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::embedding_renorm_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
