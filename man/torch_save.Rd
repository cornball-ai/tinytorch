% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_save}
\alias{torch_save}
\title{Save a torch object to a file}
\usage{
torch_save(obj, path)
}
\arguments{
\item{obj}{Object to save.}

\item{path}{File path.}
}
\value{
Invisible NULL.
}
\description{
Save a torch object to a file
}
\examples{
\donttest{
if (torch_is_installed()) {
x <- torch_randn(c(2, 3))
f <- tempfile(fileext = ".pt")
torch_save(x, f)
unlink(f)
}
}
}
