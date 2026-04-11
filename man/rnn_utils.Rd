% tinyrox says don't edit this manually, but it can't stop you!
\name{rnn_utils}
\alias{rnn_utils}
\alias{nn_utils_rnn_pack_padded_sequence}
\alias{nn_utils_rnn_pack_sequence}
\alias{nn_utils_rnn_pad_packed_sequence}
\alias{nn_utils_rnn_pad_sequence}
\title{RNN packing/padding stubs}
\usage{
nn_utils_rnn_pack_padded_sequence(input, lengths, batch_first = FALSE,
                                  enforce_sorted = TRUE)

nn_utils_rnn_pack_sequence(sequences, enforce_sorted = TRUE)

nn_utils_rnn_pad_packed_sequence(sequence, batch_first = FALSE,
                                 padding_value = 0, total_length = NULL)

nn_utils_rnn_pad_sequence(sequences, batch_first = FALSE, padding_value = 0)
}
\arguments{
\item{input}{Input tensor or list.}

\item{lengths}{Sequence lengths.}

\item{batch_first}{Logical; treat the first dim as batch.}

\item{enforce_sorted}{Logical; require sequences pre-sorted by length.}

\item{sequences}{List of tensors to pack or pad.}

\item{sequence}{A packed or padded sequence produced by the pack/pad
helpers.}

\item{padding_value}{Scalar padding value.}

\item{total_length}{Optional target padded length.}
}
\value{
Packed/padded data. Currently raises an error because
  `PackedSequence` is not implemented in tinytorch.
}
\description{
RNN packing/padding stubs
}
\examples{
\donttest{
if (torch_is_installed()) {
# Not yet implemented; will signal an error.
try(nn_utils_rnn_pack_sequence(list(torch_tensor(1:3))), silent = TRUE)
}
}

\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}

\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}

\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}

\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
