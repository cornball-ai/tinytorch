% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_get_num_threads}
\alias{torch_get_num_threads}
\title{Get Number of CPU Threads}
\usage{
torch_get_num_threads()
}
\value{
Integer number of threads used for intraop parallelism.
}
\description{
Get Number of CPU Threads
}
\examples{
\donttest{
if (torch_is_installed()) {
torch_get_num_threads()
}
}
}
