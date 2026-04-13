% tinyrox says don't edit this manually, but it can't stop you!
\name{cuda_empty_cache}
\alias{cuda_empty_cache}
\title{Release cached CUDA memory}
\usage{
cuda_empty_cache()
}
\value{
No return value, called for side effects.
}
\description{
Releases all unoccupied cached memory held by the CUDA allocator
so that it can be used by other GPU applications.
}
\examples{
\donttest{
if (torch_is_installed()) {
cuda_empty_cache()
}
}
}
