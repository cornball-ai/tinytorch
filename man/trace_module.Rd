% tinyrox says don't edit this manually, but it can't stop you!
\name{trace_module}
\alias{trace_module}
\title{Trace an nn_module for Optimized Execution}
\usage{
trace_module(
  module,
  ...,
  .optimize = TRUE,
  .fuse = TRUE,
  .backend = "auto",
  .verbose = FALSE
)
}
\arguments{
\item{module}{An nn_module callable}

\item{...}{Named example tensors matching the forward() signature}

\item{.optimize}{Logical, run optimization passes on the IR}

\item{.fuse}{Logical, compile fusion groups to kernels}

\item{.backend}{Character: "auto", "gpu", or "cpu". "auto" detects
from input tensors at call time and ariel availability.}

\item{.verbose}{Logical, print tracing info}
}
\value{
A list with:
  \item{fn}{A function that executes the traced graph}
  \item{expanded}{The expanded statement list}
  \item{params}{Named list of parameter tensors}
  \item{graph_breaks}{List of graph break descriptions}
  \item{ir}{The IR graph (if no graph breaks), or NULL}
}
\description{
Expands a module's forward() body via recursive AST rewriting,
resolves parameters and sub-modules, and optionally optimizes
through the IR pipeline.
}
