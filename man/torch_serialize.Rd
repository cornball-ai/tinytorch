% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_serialize}
\alias{torch_serialize}
\title{Serialize a torch object to a raw vector}
\usage{
torch_serialize(obj)
}
\arguments{
\item{obj}{Object to serialize.}
}
\value{
Raw vector.
}
\description{
Serialize a torch object to a raw vector
}
\examples{
\donttest{
if (torch_is_installed()) {
bytes <- torch_serialize(torch_randn(c(2, 3)))
}
}
}
