% tinyrox says don't edit this manually, but it can't stop you!
\name{is_available}
\alias{is_available}
\title{Check if tinytorch backend is available}
\usage{
is_available()
}
\value{
Logical scalar.
}
\description{
Returns TRUE if the package was compiled with libtorch support.
When FALSE, all tensor operations will error. Use this to guard
code that requires the libtorch backend.
}
\examples{
is_available()
}
