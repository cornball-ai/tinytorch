% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_set_num_interop_threads}
\alias{torch_set_num_interop_threads}
\title{Set Number of Interop Threads}
\usage{
torch_set_num_interop_threads(n)
}
\arguments{
\item{n}{Integer number of threads.}
}
\description{
Controls the number of threads used for interop parallelism
(e.g., across independent operations) in libtorch.
}
\examples{
\donttest{
if (torch_is_installed()) {
# May error if inter-op pool already initialized
try(torch_set_num_interop_threads(2L), silent = TRUE)
}
}
}
