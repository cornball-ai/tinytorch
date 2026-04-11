% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_is_vulkan_available}
\alias{torch_is_vulkan_available}
\title{Is vulkan available}
\usage{
torch_is_vulkan_available()
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::is_vulkan_available}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
