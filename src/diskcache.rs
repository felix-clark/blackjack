//! A tiny on-disk cache for expensive solver outputs, so a chosen (shoe, ruleset) does not have to be
//! re-solved from scratch every launch. The slow count-index background fill in particular is minutes of
//! CPU per shoe/ruleset; persisting it (and the base per-up-card columns) turns the app from a live
//! simulation engine into a smooth explorer once a configuration has been visited once.
//!
//! This is the one place the project leaves its (otherwise) std-only-plus-`ratatui` discipline: caching
//! needs serialization, so the cached types `derive({Serialize, Deserialize})` and we encode with
//! `bincode`. The cache is best-effort and side-channel — every operation degrades to a miss (recompute)
//! on any I/O or decode error, so a missing/corrupt/older cache never breaks correctness, only speed.
//!
//! Layout: `$XDG_CACHE_HOME/blackjack/v{SCHEMA_VERSION}/{kind}-{hash}.bin`. The schema version lives in
//! the path, so bumping it (whenever a cached type's layout or meaning changes) transparently ignores —
//! and orphans, for later GC — every older file. The hash is only a filename; the full key bytes are
//! stored in the payload and re-checked on load, so a hash collision is a miss, not a wrong answer.

use std::collections::hash_map::DefaultHasher;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;
use serde::de::DeserializeOwned;

/// Bump on any change to the on-disk layout or semantics of a cached type. Old files (under the prior
/// version's subdirectory) are then simply never read again.
const SCHEMA_VERSION: u32 = 1;

/// Process-unique suffix source for temp files, so concurrent writers (the chart workers and the
/// index-fill workers all cache in parallel) never collide on the same scratch path.
static SEQ: AtomicU64 = AtomicU64::new(0);

/// Disk caching is on by default; set `BLACKJACK_NO_CACHE` to disable (e.g. to force a clean re-solve
/// while validating the solver itself).
fn enabled() -> bool {
    env::var_os("BLACKJACK_NO_CACHE").is_none()
}

/// The versioned cache directory, or `None` if neither `XDG_CACHE_HOME` nor `HOME` is set (in which
/// case caching silently no-ops).
fn cache_root() -> Option<PathBuf> {
    let base = env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("blackjack").join(format!("v{SCHEMA_VERSION}")))
}

/// The file a `(kind, key)` maps to: the key's encoded bytes are hashed to a stable 64-bit filename.
fn path_for(kind: &str, key_bytes: &[u8]) -> Option<PathBuf> {
    let mut h = DefaultHasher::new();
    key_bytes.hash(&mut h);
    Some(cache_root()?.join(format!("{kind}-{:016x}.bin", h.finish())))
}

/// Look up a cached value. Returns `None` on a miss, a stale schema, a hash collision (stored key bytes
/// differ), or any I/O/decode error — all of which the caller handles by recomputing.
pub(crate) fn load<K: Serialize, V: DeserializeOwned>(kind: &str, key: &K) -> Option<V> {
    if !enabled() {
        return None;
    }
    let key_bytes = bincode::serialize(key).ok()?;
    let path = path_for(kind, &key_bytes)?;
    let raw = fs::read(&path).ok()?;
    let (stored_key, val): (Vec<u8>, V) = bincode::deserialize(&raw).ok()?;
    // The filename is only a hash; confirm the full key before trusting the payload.
    (stored_key == key_bytes).then_some(val)
}

/// Persist a value under `(kind, key)`. Best-effort: any failure (no cache dir, serialize error, I/O
/// error) is swallowed — the value simply will not be cached. Writes via a unique temp file + rename so
/// a concurrent reader or a crash mid-write never observes a torn file.
pub(crate) fn store<K: Serialize, V: Serialize>(kind: &str, key: &K, val: &V) {
    if !enabled() {
        return;
    }
    let Ok(key_bytes) = bincode::serialize(key) else {
        return;
    };
    let Some(path) = path_for(kind, &key_bytes) else {
        return;
    };
    let Some(dir) = path.parent() else {
        return;
    };
    if fs::create_dir_all(dir).is_err() {
        return;
    }
    let Ok(payload) = bincode::serialize(&(key_bytes, val)) else {
        return;
    };
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = path.with_extension(format!("tmp-{}-{seq}", std::process::id()));
    if fs::write(&tmp, payload).is_ok() {
        let _ = fs::rename(&tmp, &path);
    } else {
        let _ = fs::remove_file(&tmp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_then_load_roundtrips_and_distinct_keys_miss() {
        // Isolate to a throwaway cache dir (the only test touching XDG_CACHE_HOME, so the process-global
        // mutation does not race any other test).
        let dir = env::temp_dir().join(format!("blackjack-diskcache-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        // SAFETY: single test owns this env var for its duration; no other test reads it.
        unsafe {
            env::set_var("XDG_CACHE_HOME", &dir);
        }

        let key = (7u8, 3i16);
        assert!(
            load::<_, Vec<i32>>("t", &key).is_none(),
            "miss before store"
        );
        store("t", &key, &vec![1, 2, 3]);
        assert_eq!(
            load::<_, Vec<i32>>("t", &key),
            Some(vec![1, 2, 3]),
            "roundtrip"
        );
        // A different key must miss (and the full key is re-checked, so even a filename-hash collision
        // would surface as a miss rather than the wrong value).
        assert!(
            load::<_, Vec<i32>>("t", &(8u8, 3i16)).is_none(),
            "distinct key misses"
        );

        let _ = fs::remove_dir_all(&dir);
        // SAFETY: see above.
        unsafe {
            env::remove_var("XDG_CACHE_HOME");
        }
    }
}
