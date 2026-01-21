//! Debug utilities for tensor visualization.
//!
//! This module provides human-readable tensor formatting for debugging:
//!
//! - Configurable precision and width
//! - Summarization for large tensors
//! - Shape and statistics display

use super::Tensor;
use std::fmt::Write;

/// Configuration for tensor pretty-printing.
#[derive(Debug, Clone)]
pub struct PrintOptions {
    /// Number of decimal places for floats.
    pub precision: usize,
    /// Maximum elements to show per dimension edge.
    pub edge_items: usize,
    /// Threshold for summarization (show "..." if exceeded).
    pub threshold: usize,
    /// Minimum width for each element.
    pub min_width: usize,
    /// Whether to show shape header.
    pub show_shape: bool,
}

impl Default for PrintOptions {
    fn default() -> Self {
        Self {
            precision: 4,
            edge_items: 3,
            threshold: 1000,
            min_width: 8,
            show_shape: true,
        }
    }
}

impl PrintOptions {
    /// Creates options for compact output.
    #[must_use]
    pub fn compact() -> Self {
        Self {
            precision: 2,
            edge_items: 2,
            threshold: 100,
            min_width: 6,
            show_shape: false,
        }
    }

    /// Creates options for full output (no summarization).
    #[must_use]
    pub fn full() -> Self {
        Self {
            precision: 6,
            edge_items: usize::MAX,
            threshold: usize::MAX,
            min_width: 10,
            show_shape: true,
        }
    }
}

/// Pretty-prints a tensor with the given options.
pub fn format_tensor<T: std::fmt::Display>(tensor: &Tensor<T>, opts: &PrintOptions) -> String {
    let mut output = String::new();

    if opts.show_shape {
        writeln!(output, "Tensor shape: {}", tensor.shape()).unwrap();
    }

    let numel = tensor.numel();
    let summarize = numel > opts.threshold;

    match tensor.ndim() {
        0 => {
            // Scalar
            if let Some(val) = tensor.get(&[]) {
                writeln!(output, "{val}").unwrap();
            }
        }
        1 => {
            format_1d(tensor, opts, summarize, &mut output);
        }
        2 => {
            format_2d(tensor, opts, summarize, &mut output);
        }
        _ => {
            format_nd(tensor, opts, summarize, &mut output);
        }
    }

    output
}

fn format_1d<T: std::fmt::Display>(
    tensor: &Tensor<T>,
    opts: &PrintOptions,
    summarize: bool,
    output: &mut String,
) {
    let len = tensor.dims()[0];
    write!(output, "[").unwrap();

    if summarize && len > 2 * opts.edge_items {
        // Show first edge_items
        for i in 0..opts.edge_items {
            if let Some(val) = tensor.get(&[i]) {
                write!(output, "{:>width$.prec$}", val, width = opts.min_width, prec = opts.precision).unwrap();
            }
            if i < opts.edge_items - 1 {
                write!(output, ", ").unwrap();
            }
        }
        write!(output, ", ..., ").unwrap();
        // Show last edge_items
        for i in (len - opts.edge_items)..len {
            if let Some(val) = tensor.get(&[i]) {
                write!(output, "{:>width$.prec$}", val, width = opts.min_width, prec = opts.precision).unwrap();
            }
            if i < len - 1 {
                write!(output, ", ").unwrap();
            }
        }
    } else {
        for i in 0..len {
            if let Some(val) = tensor.get(&[i]) {
                write!(output, "{:>width$.prec$}", val, width = opts.min_width, prec = opts.precision).unwrap();
            }
            if i < len - 1 {
                write!(output, ", ").unwrap();
            }
        }
    }

    writeln!(output, "]").unwrap();
}

fn format_2d<T: std::fmt::Display>(
    tensor: &Tensor<T>,
    opts: &PrintOptions,
    summarize: bool,
    output: &mut String,
) {
    let rows = tensor.dims()[0];
    let cols = tensor.dims()[1];

    writeln!(output, "[").unwrap();

    let row_indices: Vec<usize> = if summarize && rows > 2 * opts.edge_items {
        let mut indices: Vec<usize> = (0..opts.edge_items).collect();
        indices.push(usize::MAX); // Marker for "..."
        indices.extend((rows - opts.edge_items)..rows);
        indices
    } else {
        (0..rows).collect()
    };

    for (idx, &row) in row_indices.iter().enumerate() {
        if row == usize::MAX {
            writeln!(output, "  ...").unwrap();
            continue;
        }

        write!(output, "  [").unwrap();

        let col_indices: Vec<usize> = if summarize && cols > 2 * opts.edge_items {
            let mut indices: Vec<usize> = (0..opts.edge_items).collect();
            indices.push(usize::MAX);
            indices.extend((cols - opts.edge_items)..cols);
            indices
        } else {
            (0..cols).collect()
        };

        for (cidx, &col) in col_indices.iter().enumerate() {
            if col == usize::MAX {
                write!(output, " ... ").unwrap();
                continue;
            }

            if let Some(val) = tensor.get(&[row, col]) {
                write!(output, "{:>width$.prec$}", val, width = opts.min_width, prec = opts.precision).unwrap();
            }

            if cidx < col_indices.len() - 1 && col_indices[cidx + 1] != usize::MAX {
                write!(output, ", ").unwrap();
            }
        }

        write!(output, "]").unwrap();
        if idx < row_indices.len() - 1 {
            writeln!(output, ",").unwrap();
        } else {
            writeln!(output).unwrap();
        }
    }

    writeln!(output, "]").unwrap();
}

fn format_nd<T: std::fmt::Display>(
    tensor: &Tensor<T>,
    opts: &PrintOptions,
    _summarize: bool,
    output: &mut String,
) {
    // For higher dimensions, show a summary
    let dims = tensor.dims();
    writeln!(output, "Tensor with {} dimensions: {:?}", dims.len(), dims).unwrap();
    writeln!(output, "Total elements: {}", tensor.numel()).unwrap();

    // Show first few elements
    write!(output, "First elements: [").unwrap();
    let show_count = opts.edge_items.min(tensor.numel());
    for i in 0..show_count {
        if let Some(val) = tensor.as_slice().get(i) {
            write!(output, "{:>width$.prec$}", val, width = opts.min_width, prec = opts.precision).unwrap();
            if i < show_count - 1 {
                write!(output, ", ").unwrap();
            }
        }
    }
    if tensor.numel() > show_count {
        write!(output, ", ...").unwrap();
    }
    writeln!(output, "]").unwrap();
}

/// Computes basic statistics for a tensor.
#[derive(Debug, Clone)]
pub struct TensorStats {
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Mean value.
    pub mean: f64,
    /// Standard deviation.
    pub std: f64,
    /// Number of elements.
    pub count: usize,
}

impl TensorStats {
    /// Computes statistics for an f32 tensor.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_f32(tensor: &Tensor<f32>) -> Self {
        let slice = tensor.as_slice();
        let count = slice.len();

        if count == 0 {
            return Self {
                min: f64::NAN,
                max: f64::NAN,
                mean: f64::NAN,
                std: f64::NAN,
                count: 0,
            };
        }

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut sum = 0.0_f64;

        for &val in slice {
            let v = f64::from(val);
            min = min.min(v);
            max = max.max(v);
            sum += v;
        }

        let mean = sum / count as f64;

        let mut var_sum = 0.0_f64;
        for &val in slice {
            let diff = f64::from(val) - mean;
            var_sum += diff * diff;
        }
        let std = (var_sum / count as f64).sqrt();

        Self {
            min,
            max,
            mean,
            std,
            count,
        }
    }
}

impl std::fmt::Display for TensorStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Stats(min={:.4}, max={:.4}, mean={:.4}, std={:.4}, n={})",
            self.min, self.max, self.mean, self.std, self.count
        )
    }
}

// Extension methods for Tensor
impl Tensor<f32> {
    /// Pretty-prints the tensor with default options.
    #[must_use]
    pub fn pretty(&self) -> String {
        format_tensor(self, &PrintOptions::default())
    }

    /// Pretty-prints the tensor with custom options.
    #[must_use]
    pub fn pretty_with(&self, opts: &PrintOptions) -> String {
        format_tensor(self, opts)
    }

    /// Computes basic statistics.
    #[must_use]
    pub fn stats(&self) -> TensorStats {
        TensorStats::from_f32(self)
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;

    #[test]
    fn test_print_1d() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let output = tensor.pretty();
        assert!(output.contains("shape: [5]"));
        assert!(output.contains("1.0"));
        assert!(output.contains("5.0"));
    }

    #[test]
    fn test_print_2d() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let output = tensor.pretty();
        assert!(output.contains("shape: [2, 3]"));
    }

    #[test]
    fn test_print_summarized() {
        // Use more than default threshold of 1000 elements
        let data: Vec<f32> = (0..1500).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![1500]).unwrap();
        let output = tensor.pretty();
        assert!(output.contains("..."), "Output should contain '...': {output}");
    }

    #[test]
    fn test_stats() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let stats = tensor.stats();

        assert!((stats.min - 1.0).abs() < 1e-6);
        assert!((stats.max - 5.0).abs() < 1e-6);
        assert!((stats.mean - 3.0).abs() < 1e-6);
        assert_eq!(stats.count, 5);
    }

    #[test]
    fn test_compact_options() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let output = tensor.pretty_with(&PrintOptions::compact());
        assert!(!output.contains("shape")); // Compact hides shape
    }
}