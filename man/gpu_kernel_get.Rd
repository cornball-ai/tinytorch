% tinyrox says don't edit this manually, but it can't stop you!
\name{gpu_kernel_get}
\alias{gpu_kernel_get}
\title{Get cached GPU kernel}
\usage{
gpu_kernel_get(ir_hash, sm_version)
}
\arguments{
\item{ir_hash}{Character hash of IR graph structure}

\item{sm_version}{Integer CUDA compute capability (e.g., 86 for sm_86)}
}
\value{
Cached PTX string or NULL if not found
}
\description{
Get cached GPU kernel
}
\examples{
gpu_kernel_get("abc123", 86)
}
