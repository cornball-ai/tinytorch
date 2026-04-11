% tinyrox says don't edit this manually, but it can't stop you!
\name{dataset}
\alias{dataset}
\title{Create a dataset}
\usage{
dataset(name = NULL, ...)
}
\arguments{
\item{name}{Dataset class name.}

\item{...}{Named methods: initialize, .getitem, .length.}
}
\value{
A dataset generator function.
}
\description{
Create a dataset
}
\examples{
\donttest{
if (torch_is_installed()) {
# See vignette("datasets") for full dataset/dataloader usage
}
}
}
