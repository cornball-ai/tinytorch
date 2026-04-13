% tinyrox says don't edit this manually, but it can't stop you!
\name{length.torch_dataset}
\alias{length.torch_dataset}
\title{Length.torch dataset}
\usage{
\method{length}{torch_dataset}(x)
}
\arguments{
\item{x}{Parameter passed to the underlying ATen operator.}
}
\value{
Integer scalar.
}
\description{
Length.torch dataset
}
\examples{
\donttest{
if (torch_is_installed()) {
# See ?dataset for dataloader usage
}
}
}
