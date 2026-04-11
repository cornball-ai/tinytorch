% tinyrox says don't edit this manually, but it can't stop you!
\name{cuda_memory_stats}
\alias{cuda_memory_stats}
\title{CUDA memory statistics from libtorch caching allocator}
\usage{
cuda_memory_stats()
}
\value{
Named numeric vector: allocated_current, allocated_peak,
  reserved_current, reserved_peak.
}
\description{
Returns allocated and reserved memory in bytes. "Allocated" is memory
actively used by tensors. "Reserved" is total memory held by the
caching allocator (allocated + cached free blocks).
}
\examples{
\donttest{
if (torch_is_installed()) {
if (cuda_is_available()) cuda_memory_stats()
}
}
}
