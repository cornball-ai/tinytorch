% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_log2}
\alias{torch_log2}
\title{torch_log2}
\usage{
torch_log2(self)
}
\arguments{
\item{self}{A `torch_tensor`.}
}
\value{
A `torch_tensor`.
}
\description{
torch_log2
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_rand(5)
a
torch_log2(a)
}
}
}
