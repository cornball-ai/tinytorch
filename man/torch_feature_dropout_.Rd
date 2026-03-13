% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_feature_dropout_}
\alias{torch_feature_dropout_}
\title{Feature dropout}
\usage{
torch_feature_dropout_(self, p, train)
}
\arguments{
\item{self}{A torch_tensor.}

\item{p}{A numeric value.}

\item{train}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::feature_dropout_}.
}
