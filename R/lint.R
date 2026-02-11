#' Lint Torch Expressions for Graph Breaks
#'
#' Analyzes an R expression and reports any operations that would cause
#' graph breaks, preventing full compilation. Provides suggestions for
#' improving performance.
#'
#' @param expr An R expression to analyze (quoted or unquoted)
#' @param suggest Logical, whether to include suggestions. Default TRUE.
#' @return Invisibly returns a list with lint results. Prints a report.
#'
#' @examples
#' \dontrun{
#' torch_lint(quote({
#'   z <- x$matmul(y)
#'   print(z$shape)
#'   z$relu()
#' }))
#' }
#'
#' @export
torch_lint <- function(expr, suggest = TRUE) {
  # Capture if not already quoted
  if (!is.call(expr) && !is.symbol(expr)) {
    expr <- substitute(expr)
  }

  walked <- walk_expr(expr)
  breaks <- find_graph_breaks(expr)
  segments <- segment_expr(expr)

  n_breaks <- length(breaks)
  n_segments <- length(segments)
  n_graph_segments <- sum(vapply(segments, function(s) s$type == "graph", logical(1)))
  n_statements <- if (walked$type == "block") walked$n_statements else 1L

  # Build report
  message("")
  message(rule("torchlang lint report"))
  message("")

  # Summary
  if (n_breaks == 0) {
    message(success("No graph breaks detected"))
    message(info(sprintf("Expression is fully compilable (%d statement%s)",
                         n_statements, if (n_statements == 1) "" else "s")))
  } else {
    message(warning_msg(sprintf("Found %d graph break%s",
                                n_breaks, if (n_breaks == 1) "" else "s")))
    message(info(sprintf("Expression will be split into %d segment%s (%d compilable)",
                         n_segments, if (n_segments == 1) "" else "s", n_graph_segments)))
  }

  # Detail each break

  if (n_breaks > 0) {
    message("")
    message(header("Graph breaks"))

    for (i in seq_along(breaks)) {
      b <- breaks[[i]]

      # Type-specific message
      msg <- switch(b$type,
        "side_effect" = " - side effect, cannot be compiled",
        "control_flow" = " - data-dependent control flow",
        "function_call" = " - unknown function (not a torch op)",
        " - causes graph break"
      )
      message(sprintf("\n  %d. %s%s", i, emphasis(b$detail), dim_text(msg)))

      # Suggestion
      if (suggest) {
        suggestion <- get_suggestion(b)
        if (!is.null(suggestion)) {
          message(sprintf("     %s %s", arrow(), suggestion))
        }
      }
    }
  }

  # Segment visualization
  if (n_segments > 1) {
    message("")
    message(header("Segment structure"))
    message("")

    for (i in seq_along(segments)) {
      seg <- segments[[i]]
      if (seg$type == "graph") {
        message(sprintf("  %s Segment %d: %d statement%s %s",
                        graph_icon(), i,
                        length(seg$statements),
                        if (length(seg$statements) == 1) "" else "s",
                        dim_text("[compilable]")))
      } else {
        message(sprintf("  %s Segment %d: %d statement%s %s",
                        break_icon(), i,
                        length(seg$statements),
                        if (length(seg$statements) == 1) "" else "s",
                        dim_text("[R code]")))
      }
    }
  }

  message("")
  message(rule())

  # Return results invisibly
  invisible(list(
    n_breaks = n_breaks,
    n_segments = n_segments,
    n_graph_segments = n_graph_segments,
    breaks = breaks,
    segments = segments,
    is_clean = n_breaks == 0
  ))
}


#' Get Suggestion for a Graph Break
#' @noRd
get_suggestion <- function(break_info) {
  detail <- break_info$detail
  type <- break_info$type

  if (type == "side_effect") {
    if (detail %in% c("print", "cat", "message")) {
      return("Remove or move outside torch_eval() for better performance")
    }
    if (detail %in% c("warning", "stop"))
      return("Consider handling errors outside the compiled region")
    if (detail %in% c("plot", "lines", "points")) {
      return("Move plotting code outside torch_eval()")
    }
    return("Side effects prevent compilation")
  }

  if (type == "control_flow") {
    if (detail == "if") {
      return("Use torch_where() for conditional tensor ops, or split into separate torch_eval() calls")
    }
    if (detail %in% c("for", "while")) {
      return("Unroll loop or use vectorized torch ops instead")
    }
  }

  if (type == "function_call") {
    return(sprintf("If '%s' operates on tensors, consider using torch ops directly", detail))
  }

  NULL
}


# --- Formatting helpers (simple, no cli dependency) ---

rule <- function(title = NULL) {
  width <- 50
  if (is.null(title)) {
    paste(rep("-", width), collapse = "")
  } else {
    n_dash <- width - nchar(title) - 2
    left <- floor(n_dash / 2)
    right <- ceiling(n_dash / 2)
    paste0(paste(rep("-", left), collapse = ""), " ", title, " ",
           paste(rep("-", right), collapse = ""))
  }
}

header <- function(text) {
  paste0(text, ":")
}

success <- function(text) {
  paste0("[OK] ", text)
}

warning_msg <- function(text) {
  paste0("[!] ", text)
}
info <- function(text) {
  paste0("    ", text)
}

emphasis <- function(text) {
  paste0("'", text, "'")
}

dim_text <- function(text) {
  text
}

arrow <- function() {
  "->"
}

graph_icon <- function() {
  "[G]"
}

break_icon <- function() {
  "[R]"
}
