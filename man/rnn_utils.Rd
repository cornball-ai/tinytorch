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

\item{...}{Additional arguments.}
}
\value{
Packed/padded data.
}
\description{
RNN packing/padding stubs
}
