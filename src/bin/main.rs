//! gguf-inspect CLI — commit 8.3.
//!
//! Accepts a GGUF model file path and prints a formatted inspection summary.
//!
//! # Usage
//!
//! ```text
//! llm <model.gguf>             # print metadata summary
//! llm <model.gguf> --tensors  # also list every tensor name/shape/dtype
//! ```

#![warn(clippy::all)]
#![warn(clippy::pedantic)]

use std::process;
use llm_engine::gguf::{GgufLoader, inspect};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // ── argument parsing (no external deps) ──────────────────────────────
    let path = args.get(1).unwrap_or_else(|| {
        eprintln!("Usage: llm <model.gguf> [--tensors]");
        process::exit(1);
    });

    if path == "--help" || path == "-h" {
        println!("Usage: llm <model.gguf> [--tensors]");
        println!();
        println!("  <model.gguf>   Path to a GGUF model file");
        println!("  --tensors      Dump all tensor names, shapes, and dtypes");
        return;
    }

    let show_tensors = args.iter().any(|a| a == "--tensors");

    // ── metadata summary ─────────────────────────────────────────────────
    let inspection = inspect(path).unwrap_or_else(|e| {
        eprintln!("Error: failed to inspect '{path}': {e}");
        process::exit(1);
    });

    print!("{inspection}");

    // ── optional tensor dump ─────────────────────────────────────────────
    if show_tensors {
        let loader = GgufLoader::open(path).unwrap_or_else(|e| {
            eprintln!("Error: failed to open '{path}': {e}");
            process::exit(1);
        });

        println!("\nTensors ({} total):", loader.tensors().len());
        println!("{:<60} {:>12}  {}", "Name", "Shape", "Dtype");
        println!("{}", "-".repeat(90));

        let mut tensors: Vec<_> = loader.tensors().iter().collect();
        tensors.sort_by(|a, b| a.name().cmp(b.name()));

        for t in tensors {
            let shape: Vec<String> = t.dims().iter().map(|d| d.to_string()).collect();
            println!(
                "{:<60} {:>12}  {}",
                t.name(),
                format!("[{}]", shape.join(", ")),
                t.dtype().name(),
            );
        }
    }
}
