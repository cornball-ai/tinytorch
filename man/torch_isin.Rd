% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_isin}
\alias{torch_isin}
\title{Isin}
\usage{
torch_isin(elements, test_elements, assume_unique = FALSE, invert = FALSE)
}
\arguments{
\item{elements}{A torch_tensor.}

\item{test_elements}{A torch_tensor.}

\item{assume_unique}{A logical value.}

\item{invert}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::isin}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
