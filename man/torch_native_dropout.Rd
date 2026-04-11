% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_native_dropout}
\alias{torch_native_dropout}
\title{Native dropout}
\usage{
torch_native_dropout(input, p, train)
}
\arguments{
\item{input}{A torch_tensor.}

\item{p}{A numeric value.}

\item{train}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::native_dropout}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
