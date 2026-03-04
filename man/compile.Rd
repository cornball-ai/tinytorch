% tinyrox says don't edit this manually, but it can't stop you!
\name{compile}
\alias{compile}
\title{Compile and Save/Load Optimized Torch Modules}
\usage{
compile(obj, ..., path = NULL, .optimize = TRUE, .fuse = TRUE, .backend = "auto")
}
\arguments{
\item{obj}{An nn_module callable or a plain R function containing
torch operations.}

\item{...}{Named example tensors matching the forward() signature
(for modules) or function formals.}

\item{path}{Optional file path (with \code{.torchlang} extension)
to save the compiled artifact. If non-NULL, the artifact is saved
and the \code{compiled_module} is returned invisibly.}

\item{.optimize}{Logical, run optimization passes on the IR
(default TRUE).}

\item{.fuse}{Logical, compile fusion groups to kernels (default TRUE).}

\item{.backend}{Character: "auto", "gpu", or "cpu". "auto" detects
from input tensors and ariel availability.}
}
\value{
A \code{compiled_module} object (invisibly if \code{path}
  is non-NULL).
}
\description{
Eagerly compiles nn_modules or plain R functions into optimized
computation graphs that can be saved to disk and reloaded without
re-tracing.
Unlike Python's \code{torch.compile()}, which compiles lazily on
first call, R's \code{body()} gives full ASTs without running
anything, so we compile eagerly and can save portable artifacts.
Compile an nn_module or Function
Traces and compiles a torch module or plain R function into an
optimized computation graph. Optionally saves the compiled artifact
to disk.
}
\examples{
\dontrun{
library(torch)

# Compile a module
model <- nn_linear(10, 5)
compiled <- compile(model, input = torch_randn(1, 10))
compiled(input = torch_randn(1, 10))

# Compile a plain function
compiled <- compile(function(x, y) x$matmul(y)$relu(),
                    x = torch_randn(3, 3), y = torch_randn(3, 3))
compiled(x = torch_randn(3, 3), y = torch_randn(3, 3))

# Save/load round-trip
compile(model, input = torch_randn(1, 10), path = "/tmp/model.torchlang")
loaded <- load("/tmp/model.torchlang")
loaded(input = torch_randn(1, 10))
}

}
