% tinyrox says don't edit this manually, but it can't stop you!
\name{dataset_subset}
\alias{dataset_subset}
\title{Subset of a dataset}
\usage{
dataset_subset(dataset, indices)
}
\arguments{
\item{dataset}{A torch_dataset.}

\item{indices}{Integer vector of indices.}
}
\value{
A dataset object.
}
\description{
Subset of a dataset
}
\examples{
\donttest{
if (torch_is_installed()) {
ds <- tensor_dataset(torch_randn(c(10, 3)))
sub <- dataset_subset(ds, indices = 1:5)
length(sub)
}
}
}
