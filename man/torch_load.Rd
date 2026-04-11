% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_load}
\alias{torch_load}
\title{Load a torch object from a file}
\usage{
torch_load(path)
}
\arguments{
\item{path}{File path.}
}
\value{
The loaded object.
}
\description{
Load a torch object from a file
}
\examples{
\donttest{
if (torch_is_installed()) {
x <- torch_randn(c(2, 3))
f <- tempfile(fileext = ".pt")
torch_save(x, f)
y <- torch_load(f)
unlink(f)
}
}
}
