% tinyrox says don't edit this manually, but it can't stop you!
\name{cuda_mem_info}
\alias{cuda_mem_info}
\title{CUDA memory info}
\usage{
cuda_mem_info()
}
\value{
Named numeric vector with elements "free" and "total".
}
\description{
Returns free and total GPU memory in bytes for the current device,
as reported by the CUDA runtime (cudaMemGetInfo).
}
