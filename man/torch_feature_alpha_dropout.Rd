% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_feature_alpha_dropout}
\alias{torch_feature_alpha_dropout}
\title{Feature alpha dropout}
\usage{
torch_feature_alpha_dropout(input, p, train)
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
Wrapper for \code{at::feature_alpha_dropout}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
