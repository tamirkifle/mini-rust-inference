//! LLM Inference Engine CLI
//!
//! Usage:
//!   llm generate --model <path> --prompt "Hello"
//!   llm inspect --model <path>
//!   llm bench --model <path>

#![warn(clippy::all)]
#![warn(clippy::pedantic)]

use llm_engine::VERSION;

fn main() {
    println!("LLM Inference Engine v{VERSION}");
    println!("Run with --help for usage information");

    // CLI implementation will be added in later milestones
    // using clap for argument parsing
}
