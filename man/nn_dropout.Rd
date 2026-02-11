% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_dropout}
\alias{nn_dropout}
\title{Dropout module (no-op in eval mode, Rtorch has no autograd)}
\usage{
nn_dropout(p = 0.5, inplace = FALSE)
}
\arguments{
\item{p}{Probability of dropping. Ignored.}

\item{inplace}{Ignored.}
}
\value{
An nn_module instance.
}
\description{
Dropout module (no-op in eval mode, Rtorch has no autograd)
}
