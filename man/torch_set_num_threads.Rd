% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_set_num_threads}
\alias{torch_set_num_threads}
\title{Set Number of CPU Threads}
\usage{
torch_set_num_threads(n)
}
\arguments{
\item{n}{Integer number of threads.}
}
\value{
No return value, called for side effects.
}
\description{
Controls the number of threads used for intraop parallelism
(MKL, OpenBLAS, etc.) in libtorch operations.
}
\examples{
\donttest{
if (torch_is_installed()) {
old <- torch_get_num_threads()
torch_set_num_threads(2L)
torch_set_num_threads(old)
}
}
}
