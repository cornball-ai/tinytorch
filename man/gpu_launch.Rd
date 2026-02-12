% tinyrox says don't edit this manually, but it can't stop you!
\name{gpu_launch}
\alias{gpu_launch}
\title{Launch a compiled GPU kernel (elementwise)}
\usage{
gpu_launch(ptx, kernel_name, inputs, output, grid, block, shared_mem = 0L)
}
\arguments{
\item{ptx}{Character, PTX assembly text.}

\item{kernel_name}{Character, kernel entry point name.}

\item{inputs}{List of torch_tensor (on CUDA device).}

\item{output}{torch_tensor, pre-allocated output (on CUDA device).}

\item{grid}{Integer vector of length 3, grid dimensions.}

\item{block}{Integer vector of length 3, block dimensions.}

\item{shared_mem}{Integer, shared memory bytes (default 0).}
}
\value{
Invisible NULL. Output is written in-place.
}
\description{
Launch a compiled GPU kernel (elementwise)
}
\keyword{internal}
