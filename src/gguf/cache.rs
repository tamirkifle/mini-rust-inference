//! Tensor caching and preloading strategies.
//!
//! This module provides mechanisms for caching extracted tensors to avoid
//! repeated dequantization overhead, and strategies for preloading tensors
//! to optimize model initialization.
//!
//! # Overview
//!
//! During inference, the same weight tensors are accessed repeatedly. For
//! quantized models, each access requires dequantization which can be
//! computationally expensive. Caching dequantized tensors trades memory
//! for computation time.
//!
//! # Cache Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    TensorCache<T>                       │
//! ├─────────────────────────────────────────────────────────┤
//! │  HashMap<String, CacheEntry<T>>                         │
//! │    - tensor: Tensor<T>                                  │
//! │    - last_access: Instant                               │
//! │    - access_count: usize                                │
//! │    - size_bytes: usize                                  │
//! ├─────────────────────────────────────────────────────────┤
//! │  Eviction: LRU when memory_used > memory_limit          │
//! │  Stats: hits, misses, evictions, memory tracking        │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Preloading Strategies
//!
//! - **Eager**: Load all tensors at startup (highest memory, fastest inference)
//! - **Lazy**: Load tensors on first access (lowest memory, variable latency)
//! - **Selective**: Preload specific patterns (balanced approach)
//!
//! # Example
//!
//! ```
//! use llm_engine::gguf::cache::{TensorCache, CacheConfig, PreloadStrategy};
//!
//! // Create cache with 1GB limit
//! let config = CacheConfig::new()
//!     .with_memory_limit(1024 * 1024 * 1024)
//!     .with_preload(PreloadStrategy::Lazy);
//!
//! let mut cache = TensorCache::<f32>::with_config(config);
//!
//! // Cache operations
//! // cache.insert("layer.0.weight", tensor);
//! // let tensor = cache.get("layer.0.weight");
//! ```

use std::collections::HashMap;
use std::time::Instant;

use crate::tensor::Tensor;

/// Configuration for tensor cache behavior.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum memory usage in bytes (0 = unlimited).
    pub memory_limit: usize,
    /// Preloading strategy.
    pub preload_strategy: PreloadStrategy,
    /// Whether to track detailed statistics.
    pub track_stats: bool,
    /// Patterns for selective preloading (glob-style).
    pub preload_patterns: Vec<String>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            memory_limit: 0, // Unlimited
            preload_strategy: PreloadStrategy::Lazy,
            track_stats: true,
            preload_patterns: Vec::new(),
        }
    }
}

impl CacheConfig {
    /// Creates a new cache configuration with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the memory limit in bytes.
    #[must_use]
    pub const fn with_memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = limit;
        self
    }

    /// Sets the memory limit in megabytes.
    #[must_use]
    pub const fn with_memory_limit_mb(mut self, limit_mb: usize) -> Self {
        self.memory_limit = limit_mb * 1024 * 1024;
        self
    }

    /// Sets the memory limit in gigabytes.
    #[must_use]
    pub const fn with_memory_limit_gb(mut self, limit_gb: usize) -> Self {
        self.memory_limit = limit_gb * 1024 * 1024 * 1024;
        self
    }

    /// Sets the preloading strategy.
    #[must_use]
    pub const fn with_preload(mut self, strategy: PreloadStrategy) -> Self {
        self.preload_strategy = strategy;
        self
    }

    /// Enables or disables statistics tracking.
    #[must_use]
    pub const fn with_stats(mut self, enabled: bool) -> Self {
        self.track_stats = enabled;
        self
    }

    /// Adds patterns for selective preloading.
    ///
    /// Patterns support simple glob matching:
    /// - `*` matches any sequence of characters
    /// - `?` matches any single character
    #[must_use]
    pub fn with_preload_patterns(mut self, patterns: Vec<String>) -> Self {
        self.preload_patterns = patterns;
        self
    }
}

/// Strategy for preloading tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PreloadStrategy {
    /// Load all tensors at startup.
    ///
    /// Pros: Fastest inference, predictable memory usage
    /// Cons: Slow startup, highest memory consumption
    Eager,

    /// Load tensors on first access (default).
    ///
    /// Pros: Fast startup, memory efficient
    /// Cons: First access latency, unpredictable memory growth
    #[default]
    Lazy,

    /// Preload tensors matching specific patterns.
    ///
    /// Pros: Balance between startup time and inference speed
    /// Cons: Requires knowledge of access patterns
    Selective,
}

/// Statistics for cache performance monitoring.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: usize,
    /// Number of cache misses.
    pub misses: usize,
    /// Number of tensors evicted.
    pub evictions: usize,
    /// Number of tensors currently cached.
    pub entries: usize,
    /// Total memory used by cached tensors (bytes).
    pub memory_used: usize,
    /// Peak memory usage (bytes).
    pub peak_memory: usize,
    /// Number of insert operations.
    pub inserts: usize,
}

impl CacheStats {
    /// Returns the cache hit ratio (0.0 to 1.0).
    #[must_use]
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Returns memory used in megabytes.
    #[must_use]
    pub fn memory_used_mb(&self) -> f64 {
        self.memory_used as f64 / (1024.0 * 1024.0)
    }

    /// Returns peak memory in megabytes.
    #[must_use]
    pub fn peak_memory_mb(&self) -> f64 {
        self.peak_memory as f64 / (1024.0 * 1024.0)
    }

    /// Resets all statistics except current entries and memory.
    pub fn reset_counters(&mut self) {
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
        self.inserts = 0;
        // Don't reset entries, memory_used, peak_memory
    }
}

/// Entry in the tensor cache.
#[derive(Debug)]
struct CacheEntry<T> {
    /// The cached tensor.
    tensor: Tensor<T>,
    /// When this entry was last accessed.
    last_access: Instant,
    /// Number of times this entry was accessed.
    access_count: usize,
    /// Size of this tensor in bytes.
    size_bytes: usize,
}

impl<T> CacheEntry<T> {
    fn new(tensor: Tensor<T>, size_bytes: usize) -> Self {
        Self {
            tensor,
            last_access: Instant::now(),
            access_count: 1,
            size_bytes,
        }
    }

    fn touch(&mut self) {
        self.last_access = Instant::now();
        self.access_count += 1;
    }
}

/// LRU cache for extracted tensors.
///
/// Caches dequantized tensors to avoid repeated extraction overhead.
/// Uses LRU (Least Recently Used) eviction when memory limit is reached.
#[derive(Debug)]
pub struct TensorCache<T> {
    /// Cached tensors indexed by name.
    entries: HashMap<String, CacheEntry<T>>,
    /// Cache configuration.
    config: CacheConfig,
    /// Cache statistics.
    stats: CacheStats,
}

impl<T: Clone> Default for TensorCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> TensorCache<T> {
    /// Creates a new empty cache with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            config: CacheConfig::default(),
            stats: CacheStats::default(),
        }
    }

    /// Creates a cache with the specified configuration.
    #[must_use]
    pub fn with_config(config: CacheConfig) -> Self {
        Self {
            entries: HashMap::new(),
            config,
            stats: CacheStats::default(),
        }
    }

    /// Returns the cache configuration.
    #[must_use]
    pub const fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Returns current cache statistics.
    #[must_use]
    pub const fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Returns the number of cached tensors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the total memory used by cached tensors.
    #[must_use]
    pub const fn memory_used(&self) -> usize {
        self.stats.memory_used
    }

    /// Checks if a tensor is cached.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Gets a tensor from the cache, updating access time.
    ///
    /// Returns `None` if the tensor is not cached.
    pub fn get(&mut self, name: &str) -> Option<&Tensor<T>> {
        if let Some(entry) = self.entries.get_mut(name) {
            entry.touch();
            if self.config.track_stats {
                self.stats.hits += 1;
            }
            Some(&entry.tensor)
        } else {
            if self.config.track_stats {
                self.stats.misses += 1;
            }
            None
        }
    }

    /// Gets a tensor from the cache without updating access time.
    ///
    /// Useful for inspection without affecting LRU order.
    #[must_use]
    pub fn peek(&self, name: &str) -> Option<&Tensor<T>> {
        self.entries.get(name).map(|e| &e.tensor)
    }

    /// Inserts a tensor into the cache.
    ///
    /// If memory limit is exceeded, evicts LRU entries until there's room.
    /// Returns `true` if the tensor was inserted, `false` if it was too large.
    pub fn insert(&mut self, name: impl Into<String>, tensor: Tensor<T>) -> bool {
        let name = name.into();
        let size_bytes = tensor.as_slice().len() * std::mem::size_of::<T>();

        // Check if single tensor exceeds limit
        if self.config.memory_limit > 0 && size_bytes > self.config.memory_limit {
            return false;
        }

        // Remove existing entry if present
        if let Some(old_entry) = self.entries.remove(&name) {
            self.stats.memory_used = self.stats.memory_used.saturating_sub(old_entry.size_bytes);
        }

        // Evict until we have room
        if self.config.memory_limit > 0 {
            while self.stats.memory_used + size_bytes > self.config.memory_limit {
                if !self.evict_lru() {
                    // No more entries to evict
                    return false;
                }
            }
        }

        // Insert new entry
        let entry = CacheEntry::new(tensor, size_bytes);
        self.entries.insert(name, entry);

        // Update stats
        self.stats.memory_used += size_bytes;
        self.stats.entries = self.entries.len();
        self.stats.inserts += 1;

        if self.stats.memory_used > self.stats.peak_memory {
            self.stats.peak_memory = self.stats.memory_used;
        }

        true
    }

    /// Removes a tensor from the cache.
    ///
    /// Returns the removed tensor if it existed.
    pub fn remove(&mut self, name: &str) -> Option<Tensor<T>> {
        if let Some(entry) = self.entries.remove(name) {
            self.stats.memory_used = self.stats.memory_used.saturating_sub(entry.size_bytes);
            self.stats.entries = self.entries.len();
            Some(entry.tensor)
        } else {
            None
        }
    }

    /// Clears all cached tensors.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.stats.memory_used = 0;
        self.stats.entries = 0;
    }

    /// Evicts the least recently used entry.
    ///
    /// Returns `true` if an entry was evicted, `false` if cache was empty.
    fn evict_lru(&mut self) -> bool {
        // Find LRU entry
        let lru_key = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(key, _)| key.clone());

        if let Some(key) = lru_key {
            if let Some(entry) = self.entries.remove(&key) {
                self.stats.memory_used = self.stats.memory_used.saturating_sub(entry.size_bytes);
                self.stats.entries = self.entries.len();
                self.stats.evictions += 1;
                return true;
            }
        }

        false
    }

    /// Evicts entries until memory usage is below the target.
    ///
    /// Returns the number of entries evicted.
    pub fn evict_to_size(&mut self, target_bytes: usize) -> usize {
        let mut evicted = 0;
        while self.stats.memory_used > target_bytes && self.evict_lru() {
            evicted += 1;
        }
        evicted
    }

    /// Returns names of all cached tensors.
    pub fn cached_names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// Returns an iterator over cached tensor names and their sizes.
    pub fn entries_info(&self) -> impl Iterator<Item = (&str, usize, usize)> {
        self.entries
            .iter()
            .map(|(name, entry)| (name.as_str(), entry.size_bytes, entry.access_count))
    }
}

/// Checks if a name matches a glob pattern.
///
/// Supports:
/// - `*` matches any sequence of characters
/// - `?` matches any single character
#[must_use]
pub fn matches_pattern(name: &str, pattern: &str) -> bool {
    let mut name_chars = name.chars().peekable();
    let mut pattern_chars = pattern.chars().peekable();

    while let Some(p) = pattern_chars.next() {
        match p {
            '*' => {
                // Match any sequence
                if pattern_chars.peek().is_none() {
                    // Trailing * matches everything
                    return true;
                }
                // Try matching remaining pattern at each position
                let remaining_pattern: String = pattern_chars.collect();
                loop {
                    let remaining_name: String = name_chars.clone().collect();
                    if matches_pattern(&remaining_name, &remaining_pattern) {
                        return true;
                    }
                    if name_chars.next().is_none() {
                        return false;
                    }
                }
            }
            '?' => {
                // Match any single character
                if name_chars.next().is_none() {
                    return false;
                }
            }
            c => {
                // Match literal character
                if name_chars.next() != Some(c) {
                    return false;
                }
            }
        }
    }

    // Pattern consumed, name should also be consumed
    name_chars.next().is_none()
}

/// Returns true if the name matches any of the patterns.
#[must_use]
pub fn matches_any_pattern(name: &str, patterns: &[String]) -> bool {
    patterns.iter().any(|p| matches_pattern(name, p))
}

/// Builder for preloading tensors based on strategy.
#[derive(Debug)]
pub struct PreloadBuilder {
    /// Names of tensors to preload.
    names: Vec<String>,
    /// Strategy being used.
    strategy: PreloadStrategy,
}

impl PreloadBuilder {
    /// Creates a new preload builder.
    #[must_use]
    pub fn new(strategy: PreloadStrategy) -> Self {
        Self {
            names: Vec::new(),
            strategy,
        }
    }

    /// Adds tensor names to preload.
    pub fn add_names(&mut self, names: impl IntoIterator<Item = String>) {
        self.names.extend(names);
    }

    /// Filters names based on patterns (for selective strategy).
    pub fn filter_by_patterns(&mut self, patterns: &[String]) {
        if !patterns.is_empty() {
            self.names
                .retain(|name| matches_any_pattern(name, patterns));
        }
    }

    /// Returns the names to preload.
    #[must_use]
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Returns the preload strategy.
    #[must_use]
    pub const fn strategy(&self) -> PreloadStrategy {
        self.strategy
    }

    /// Builds the list of names to preload based on strategy.
    ///
    /// # Arguments
    ///
    /// * `all_names` - All available tensor names
    /// * `patterns` - Patterns for selective preloading
    #[must_use]
    pub fn build(strategy: PreloadStrategy, all_names: &[String], patterns: &[String]) -> Self {
        let mut builder = Self::new(strategy);

        match strategy {
            PreloadStrategy::Eager => {
                builder.names = all_names.to_vec();
            }
            PreloadStrategy::Lazy => {
                // Don't preload anything
            }
            PreloadStrategy::Selective => {
                builder.names = all_names.to_vec();
                builder.filter_by_patterns(patterns);
            }
        }

        builder
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    fn make_tensor(size: usize) -> Tensor<f32> {
        Tensor::zeros(vec![size])
    }

    #[test]
    fn test_cache_basic_operations() {
        let mut cache = TensorCache::<f32>::new();

        // Insert
        let tensor = make_tensor(100);
        assert!(cache.insert("test", tensor.clone()));
        assert_eq!(cache.len(), 1);
        assert!(cache.contains("test"));

        // Get
        let retrieved = cache.get("test").unwrap();
        assert_eq!(retrieved.numel(), 100);
        assert_eq!(cache.stats().hits, 1);

        // Miss
        assert!(cache.get("nonexistent").is_none());
        assert_eq!(cache.stats().misses, 1);

        // Remove
        let removed = cache.remove("test").unwrap();
        assert_eq!(removed.numel(), 100);
        assert!(!cache.contains("test"));
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_memory_limit() {
        let config = CacheConfig::new().with_memory_limit(1000); // 1000 bytes
        let mut cache = TensorCache::<f32>::with_config(config);

        // Insert tensors that fit
        let tensor1 = make_tensor(50); // 200 bytes
        let tensor2 = make_tensor(50); // 200 bytes
        assert!(cache.insert("t1", tensor1));
        assert!(cache.insert("t2", tensor2));
        assert_eq!(cache.len(), 2);

        // Insert tensor that requires eviction
        let tensor3 = make_tensor(200); // 800 bytes
        assert!(cache.insert("t3", tensor3));
        // Should have evicted t1 (LRU)
        assert!(!cache.contains("t1"));
        assert!(cache.contains("t2") || cache.contains("t3"));

        // Tensor too large for cache
        let huge = make_tensor(500); // 2000 bytes > limit
        assert!(!cache.insert("huge", huge));
    }

    #[test]
    fn test_cache_lru_eviction() {
        // 50 elements * 4 bytes = 200 bytes per tensor
        // With 3 tensors = 600 bytes, limit of 799 means inserting 4th requires eviction
        let config = CacheConfig::new().with_memory_limit(799);
        let mut cache = TensorCache::<f32>::with_config(config);

        // Insert 3 tensors (600 bytes total)
        cache.insert("t1", make_tensor(50)); // 200 bytes
        cache.insert("t2", make_tensor(50)); // 200 bytes
        cache.insert("t3", make_tensor(50)); // 200 bytes

        // Access t1 to make it recently used
        // Access order now: t1 is MRU, t2/t3 are candidates for eviction
        cache.get("t1");

        // Insert t4, should evict one of t2 or t3 (both are LRU candidates)
        cache.insert("t4", make_tensor(50));

        // t1 was recently accessed, should NOT be evicted
        assert!(cache.contains("t1"), "t1 should be retained (recently accessed)");
        // t4 was just inserted, should be present
        assert!(cache.contains("t4"), "t4 should be present (just inserted)");
        // Either t2 or t3 should be evicted (both are equally LRU)
        assert!(
            !cache.contains("t2") || !cache.contains("t3"),
            "Either t2 or t3 should be evicted"
        );
        // Cache should have 3 entries after eviction
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = TensorCache::<f32>::new();

        cache.insert("t1", make_tensor(100));
        cache.insert("t2", make_tensor(100));

        cache.get("t1");
        cache.get("t1");
        cache.get("t2");
        cache.get("nonexistent");

        let stats = cache.stats();
        assert_eq!(stats.hits, 3);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.inserts, 2);
        assert_eq!(stats.entries, 2);
        assert!((stats.hit_ratio() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = TensorCache::<f32>::new();

        cache.insert("t1", make_tensor(100));
        cache.insert("t2", make_tensor(100));

        cache.clear();

        assert!(cache.is_empty());
        assert_eq!(cache.memory_used(), 0);
    }

    #[test]
    fn test_cache_peek() {
        let mut cache = TensorCache::<f32>::new();
        cache.insert("t1", make_tensor(100));

        // Peek doesn't update stats
        let _tensor = cache.peek("t1").unwrap();
        assert_eq!(cache.stats().hits, 0);

        // Get does update stats
        let _tensor = cache.get("t1").unwrap();
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_cache_config_builder() {
        let config = CacheConfig::new()
            .with_memory_limit_gb(2)
            .with_preload(PreloadStrategy::Eager)
            .with_stats(false)
            .with_preload_patterns(vec!["layer.*".to_string()]);

        assert_eq!(config.memory_limit, 2 * 1024 * 1024 * 1024);
        assert_eq!(config.preload_strategy, PreloadStrategy::Eager);
        assert!(!config.track_stats);
        assert_eq!(config.preload_patterns.len(), 1);
    }

    #[test]
    fn test_matches_pattern() {
        // Exact match
        assert!(matches_pattern("test", "test"));
        assert!(!matches_pattern("test", "test2"));

        // Wildcard *
        assert!(matches_pattern("layer.0.weight", "layer.*"));
        assert!(matches_pattern("layer.0.weight", "*.weight"));
        assert!(matches_pattern("layer.0.weight", "*"));
        assert!(matches_pattern("abc", "a*c"));
        assert!(!matches_pattern("abd", "a*c"));

        // Single char ?
        assert!(matches_pattern("test", "t?st"));
        assert!(matches_pattern("tast", "t?st"));
        assert!(!matches_pattern("test", "t??st"));

        // Combined
        assert!(matches_pattern("layer.10.attn.weight", "layer.*.attn.*"));
        assert!(!matches_pattern("layer.10.mlp.weight", "layer.*.attn.*"));
    }

    #[test]
    fn test_matches_any_pattern() {
        let patterns = vec!["layer.*.weight".to_string(), "embed.*".to_string()];

        assert!(matches_any_pattern("layer.0.weight", &patterns));
        assert!(matches_any_pattern("embed.tokens", &patterns));
        assert!(!matches_any_pattern("output.bias", &patterns));
    }

    #[test]
    fn test_preload_builder_eager() {
        let all_names: Vec<String> = vec!["t1".into(), "t2".into(), "t3".into()];
        let builder = PreloadBuilder::build(PreloadStrategy::Eager, &all_names, &[]);

        assert_eq!(builder.names().len(), 3);
    }

    #[test]
    fn test_preload_builder_lazy() {
        let all_names: Vec<String> = vec!["t1".into(), "t2".into(), "t3".into()];
        let builder = PreloadBuilder::build(PreloadStrategy::Lazy, &all_names, &[]);

        assert!(builder.names().is_empty());
    }

    #[test]
    fn test_preload_builder_selective() {
        let all_names: Vec<String> = vec![
            "layer.0.weight".into(),
            "layer.0.bias".into(),
            "layer.1.weight".into(),
            "embed.weight".into(),
        ];
        let patterns = vec!["*.weight".to_string()];
        let builder = PreloadBuilder::build(PreloadStrategy::Selective, &all_names, &patterns);

        assert_eq!(builder.names().len(), 3);
        assert!(builder.names().contains(&"layer.0.weight".to_string()));
        assert!(builder.names().contains(&"layer.1.weight".to_string()));
        assert!(builder.names().contains(&"embed.weight".to_string()));
        assert!(!builder.names().contains(&"layer.0.bias".to_string()));
    }

    #[test]
    fn test_evict_to_size() {
        let mut cache = TensorCache::<f32>::new();

        // Insert several tensors (100 elements * 4 bytes = 400 bytes each)
        for i in 0..5 {
            cache.insert(format!("t{i}"), make_tensor(100));
        }

        assert_eq!(cache.len(), 5);
        assert_eq!(cache.memory_used(), 2000);

        // Evict to 1000 bytes
        let evicted = cache.evict_to_size(1000);
        assert!(evicted >= 2);
        assert!(cache.memory_used() <= 1000);
    }

    #[test]
    fn test_entries_info() {
        let mut cache = TensorCache::<f32>::new();
        cache.insert("t1", make_tensor(50));
        cache.insert("t2", make_tensor(100));

        cache.get("t1");
        cache.get("t1");

        let info: Vec<_> = cache.entries_info().collect();
        assert_eq!(info.len(), 2);

        // Find t1 and check access count
        let t1_info = info.iter().find(|(name, _, _)| *name == "t1").unwrap();
        assert_eq!(t1_info.2, 3); // 1 insert + 2 gets
    }
}