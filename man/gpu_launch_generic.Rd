% tinyrox says don't edit this manually, but it can't stop you!
\name{gpu_launch_generic}
\alias{gpu_launch_generic}
\title{Launch a compiled GPU kernel (generic)}
\usage{
gpu_launch_generic(ptx, kernel_name, tensors, scalars, grid, block,
                   shared_mem = 0L)
}
\arguments{
\item{ptx}{Character, PTX assembly text.}

\item{kernel_name}{Character, kernel entry point name.}

\item{tensors}{List of torch_tensor (on CUDA device).}

\item{scalars}{List of scalar values (integer or numeric).}

\item{grid}{Integer vector of length 3, grid dimensions.}

\item{block}{Integer vector of length 3, block dimensions.}

\item{shared_mem}{Integer, shared memory bytes (default 0).}
}
\value{
Invisible NULL.
}
\description{
Flexible launcher that takes a list of tensor arguments and a list of
scalar arguments. Args are packed in order: [tensor_ptrs..., scalars...].
Scalars can be integer (passed as i32) or numeric (passed as f32).
}
\keyword{internal}
