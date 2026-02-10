% tinyrox says don't edit this manually, but it can't stop you!
\name{discover_modules}
\alias{discover_modules}
\title{Module Discovery and Automatic Tracing}
\usage{
discover_modules(module, max_depth = 10L)
}
\arguments{
\item{module}{An nn_module instance}

\item{max_depth}{Maximum recursion depth (default 10)}
}
\value{
A list of module descriptors, each with:
  \item{path}{Dot-separated path from root (e.g., "layers.1.mlp")}
  \item{class}{Module class name}
  \item{n_params}{Number of parameters}
  \item{forward_args}{Formal argument names of forward()}
  \item{module}{Live reference to the nn_module}
}
\description{
Walks an nn_module's sub-module tree and reports which modules
can be traced by torchlang. Works on any torch-dependent package
without the user needing to know the model's internals.
Walk an nn_module's Sub-module Tree
Recursively discovers all sub-modules in an nn_module instance.
Returns a flat list of module info: path, class, parameter count,
forward() argument names, and the live module reference.
}
