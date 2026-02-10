% tinyrox says don't edit this manually, but it can't stop you!
\name{gpu_kernel_set}
\alias{gpu_kernel_set}
\title{Set cached GPU kernel}
\usage{
gpu_kernel_set(ir_hash, sm_version, ptx)
}
\arguments{
\item{ir_hash}{Character hash of IR graph structure}

\item{sm_version}{Integer CUDA compute capability}

\item{ptx}{Character PTX assembly code}
}
\description{
Set cached GPU kernel
}
