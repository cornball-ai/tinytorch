% tinyrox says don't edit this manually, but it can't stop you!
\name{gpu_launch_reduction}
\alias{gpu_launch_reduction}
\title{Launch a compiled GPU reduction kernel}
\usage{
gpu_launch_reduction(ptx, kernel_name, input, output, n_elements, grid, block,
                     shared_mem = 0L)
}
\arguments{
\item{ptx}{Character, PTX assembly text.}

\item{kernel_name}{Character, kernel entry point name.}

\item{input}{torch_tensor, single input (on CUDA device).}

\item{output}{torch_tensor, pre-allocated output (on CUDA device).}

\item{n_elements}{Integer, total elements per reduction.}

\item{grid}{Integer vector of length 3, grid dimensions.}

\item{block}{Integer vector of length 3, block dimensions.}

\item{shared_mem}{Integer, shared memory bytes.}
}
\value{
Invisible NULL. Output is written in-place.
}
\description{
Launch a compiled GPU reduction kernel
}
\keyword{internal}
