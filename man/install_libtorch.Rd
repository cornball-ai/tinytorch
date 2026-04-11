% tinyrox says don't edit this manually, but it can't stop you!
\name{install_libtorch}
\alias{install_libtorch}
\title{Install libtorch}
\usage{
install_libtorch(path = "~/.local/lib/libtorch", version = "2.11.0")
}
\arguments{
\item{path}{Installation directory. Default: ~/.local/lib/libtorch}

\item{version}{Pinned libtorch version. Updated deliberately with each release.}
}
\value{
The installation path (invisibly).
}
\description{
Downloads and installs a pinned release of libtorch (PyTorch C++ distribution).
After installation, set LIBTORCH_HOME and reinstall tinytorch to enable the backend.
}
\examples{
\donttest{
if (torch_is_installed()) {
install_libtorch()
}
}
}
