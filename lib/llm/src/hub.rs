// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::OsStr;
use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};
use std::{env, fs};

use anyhow::Context;
use hf_hub::Cache;
use modelexpress_client::{
    Client as MxClient, ClientConfig as MxClientConfig, ModelProvider as MxModelProvider,
};
use modelexpress_common::{cache as mx_cache, download as mx, envs as mx_envs};

use dynamo_runtime::config::environment_names::model as env_model;

const NGC_URI_PREFIX: &str = "ngc://";

fn provider_for(model_name: &str) -> MxModelProvider {
    if model_name.starts_with(NGC_URI_PREFIX) {
        MxModelProvider::Ngc
    } else {
        MxModelProvider::HuggingFace
    }
}

/// Check if a model is already cached locally with the files needed to serve it.
/// Returns the path to the cached model directory if found, None otherwise.
///
/// For tokenizer-only downloads (ignore_weights=true), we check for config.json and
/// tokenizer files. For full downloads, we also require weight files to be present.
fn get_cached_model_path(
    model_name: &str,
    provider: MxModelProvider,
    ignore_weights: bool,
) -> Option<PathBuf> {
    let cache_dir = get_model_express_cache_dir(provider);
    match provider {
        MxModelProvider::HuggingFace => {
            get_cached_model_path_in(model_name, ignore_weights, cache_dir)
        }
        _ => get_cached_provider_model_path_in(model_name, provider, ignore_weights, &cache_dir),
    }
}

fn get_cached_model_path_in(
    model_name: &str,
    ignore_weights: bool,
    cache_dir: PathBuf,
) -> Option<PathBuf> {
    let cache = Cache::new(cache_dir);
    let snapshot_path = cache
        .model(model_name.to_string())
        .get("config.json")?
        .parent()?
        .to_path_buf();
    cached_model_dir(model_name, snapshot_path, ignore_weights)
}

/// Look up a model in a non-HuggingFace provider's cache. ModelExpress owns the layout.
fn get_cached_provider_model_path_in(
    model_name: &str,
    provider: MxModelProvider,
    ignore_weights: bool,
    cache_dir: &Path,
) -> Option<PathBuf> {
    let model_dir = mx_cache::resolve_model_path(cache_dir, provider, model_name, None).ok()?;
    cached_model_dir(model_name, model_dir, ignore_weights)
}

fn cached_model_dir(model_name: &str, dir: PathBuf, ignore_weights: bool) -> Option<PathBuf> {
    if !has_required_model_files(&dir, ignore_weights) {
        return None;
    }
    tracing::info!("Found cached model '{model_name}' at {dir:?}, skipping download");
    Some(dir)
}

/// Require model config and tokenizer files, plus weights unless explicitly skipped.
fn has_required_model_files(dir: &Path, ignore_weights: bool) -> bool {
    let has = |filename: &str| dir.join(filename).is_file();

    if !has("config.json") {
        return false;
    }

    // Check for tokenizer files (at least one must exist). Only count
    // artifacts that ``ModelDeploymentCard::TokenizerKind::from_disk`` can
    // actually load -- ``tokenizer_config.json`` is metadata describing the
    // tokenizer and cannot be used on its own, so a snapshot with only
    // ``config.json`` + ``tokenizer_config.json`` would fall through to a
    // download even though the cache appears "populated".
    let has_tokenizer = has("tokenizer.json") || has("tiktoken.model") || has_tiktoken_file(dir);
    if !has_tokenizer {
        return false;
    }

    // For full downloads, check for weight files. When an index file is present,
    // verify the shard files it references are also cached — an index without its
    // shards is an incomplete cache that should fall through to download.
    ignore_weights
        || has("model.safetensors")
        || has("pytorch_model.bin")
        || shard_files_present(&dir.join("model.safetensors.index.json"))
        || shard_files_present(&dir.join("pytorch_model.bin.index.json"))
}

/// Check if the snapshot directory contains any `*.tiktoken` file (e.g. `qwen.tiktoken`).
fn has_tiktoken_file(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .any(|e| {
            let path = e.path();
            path.extension().is_some_and(|ext| ext == "tiktoken") && path.is_file()
        })
}

/// For a sharded-weights index file (e.g. `model.safetensors.index.json`), verify
/// that every shard file it references is present in the same snapshot directory.
/// Returns false on parse error, missing weight_map, empty weight_map, or any
/// missing shard file.
fn shard_files_present(index_path: &Path) -> bool {
    let Some(snapshot_dir) = index_path.parent() else {
        return false;
    };
    let Ok(contents) = std::fs::read_to_string(index_path) else {
        return false;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&contents) else {
        return false;
    };
    let Some(weight_map) = value.get("weight_map").and_then(|v| v.as_object()) else {
        return false;
    };
    let shards: std::collections::HashSet<&str> =
        weight_map.values().filter_map(|v| v.as_str()).collect();
    if shards.is_empty() {
        return false;
    }
    shards.iter().all(|s| snapshot_dir.join(s).is_file())
}

/// Check if offline mode is enabled via HF_HUB_OFFLINE environment variable.
fn is_offline_mode() -> bool {
    dynamo_runtime::config::env_is_truthy(env_model::huggingface::HF_HUB_OFFLINE)
}

/// Check if shared-storage mode is disabled via MODEL_EXPRESS_NO_SHARED_STORAGE.
/// When true, the Model Express client streams files from the server over gRPC
/// instead of relying on a shared filesystem path. This is required when the
/// server and worker pods do not share a filesystem (e.g. RWO PVCs, cross-namespace
/// deployments).
fn is_no_shared_storage() -> bool {
    dynamo_runtime::config::env_is_truthy(env_model::model_express::MODEL_EXPRESS_NO_SHARED_STORAGE)
}

/// Download a model using ModelExpress client. The client first requests for the model
/// from the server and fallbacks to direct download in case of server failure.
/// `ngc://` names are downloaded from NGC; all other names are Hugging Face repos.
/// If ignore_weights is true, model weight files will be skipped
/// Returns the path to the model files
///
/// If the model is already cached locally with the required files, returns the cached
/// path without making any network calls, regardless of HF_HUB_OFFLINE.
pub async fn from_hf(name: impl AsRef<Path>, ignore_weights: bool) -> anyhow::Result<PathBuf> {
    let name = name.as_ref();
    let model_name = name.display().to_string();
    let provider = provider_for(&model_name);

    // Cache-first in all modes: if the snapshot is already on disk with the files we
    // need, return it without touching the network.
    if let Some(cached_path) = get_cached_model_path(&model_name, provider, ignore_weights) {
        return Ok(cached_path);
    }

    let offline = is_offline_mode();
    if offline {
        tracing::warn!(
            "Offline mode enabled but model '{model_name}' not found in cache, attempting download anyway"
        );
    }

    let config = mx_client_config(provider);
    let hf_stream_cache = (!config.cache.shared_storage
        && MxModelProvider::resolve_provider_for_model_name(&model_name, provider)
            == MxModelProvider::HuggingFace)
        .then(|| get_model_express_cache_dir(provider));
    let result = match mx_download_from_server(
        &model_name,
        provider,
        ignore_weights,
        config,
        hf_stream_cache.as_deref(),
    )
    .await
    {
        Ok(path) => Ok(path),
        Err(server_error) => {
            tracing::warn!(%model_name, error = %server_error, "ModelExpress fetch failed, falling back to direct download");
            mx_download_direct(&model_name, provider, ignore_weights)
                .await
                .and_then(|path| {
                    // The offline provider can resolve a snapshot without checking its files.
                    if hf_stream_cache.is_some() && mx_envs::hf_offline() {
                        anyhow::ensure!(
                            has_required_model_files(&path, ignore_weights),
                            "Cached Hugging Face snapshot at {} is incomplete",
                            path.display()
                        );
                    }
                    Ok(path)
                })
                .map_err(|fallback_error| {
                    anyhow::anyhow!(
                        "ModelExpress fetch failed for '{model_name}': {server_error:#}. Direct fallback failed: {fallback_error:#}"
                    )
                })
        }
    };

    match result {
        Ok(path) => {
            tracing::info!("ModelExpress download completed successfully for model: {model_name}");
            Ok(path)
        }
        Err(error) => {
            tracing::warn!(%model_name, ?error, "Model download failed");
            Err(error)
        }
    }
}

fn mx_client_config(provider: MxModelProvider) -> MxClientConfig {
    let mut config = MxClientConfig::default();
    if is_no_shared_storage() {
        config.cache.shared_storage = false;
    }
    if provider == MxModelProvider::Ngc {
        config.cache.local_path = get_model_express_cache_dir(provider);
    }
    if let Ok(endpoint) = env::var(env_model::model_express::MODEL_EXPRESS_URL) {
        config = config.with_endpoint(endpoint);
    }
    config
}

async fn mx_download_from_server(
    model_name: &str,
    provider: MxModelProvider,
    ignore_weights: bool,
    mut config: MxClientConfig,
    hf_stream_cache: Option<&Path>,
) -> anyhow::Result<PathBuf> {
    let staging = if let Some(cache) = hf_stream_cache {
        tokio::fs::create_dir_all(cache)
            .await
            .with_context(|| format!("Failed to create Hugging Face cache {}", cache.display()))?;
        // ModelExpress writes directly to final filenames. Keep the stream outside
        // the snapshot tree until the client has accepted every file.
        let staging = tempfile::Builder::new()
            .prefix(".mx-")
            .tempdir_in(cache)
            .with_context(|| format!("Failed to stage a download in {}", cache.display()))?;
        config.cache.local_path = staging.path().to_path_buf();
        Some((staging, cache.to_path_buf()))
    } else {
        None
    };

    let mut client = MxClient::new(config).await?;
    tracing::info!("Successfully connected to ModelExpress server");
    let download = client
        .request_model_revision(model_name, provider, ignore_weights, None)
        .await?;
    tracing::info!("Server download succeeded for model: {model_name}");

    if let Some((staging, cache)) = staging {
        let snapshot = download
            .path
            .context("ModelExpress returned no streamed snapshot")?;
        return tokio::task::spawn_blocking(move || {
            publish_streamed_hf_snapshot(
                staging.path(),
                &snapshot,
                &cache,
                download.resolved_revision.as_deref(),
            )
        })
        .await?;
    }

    client.get_model_path(model_name, provider).await
}

fn publish_streamed_hf_snapshot(
    staging: &Path,
    snapshot: &Path,
    cache: &Path,
    resolved_revision: Option<&str>,
) -> anyhow::Result<PathBuf> {
    let relative = snapshot.strip_prefix(staging)?;
    let components: Vec<_> = relative.components().collect();
    let (repo, revision) = match components.as_slice() {
        [
            Component::Normal(repo),
            Component::Normal(dir),
            Component::Normal(revision),
        ] if *dir == "snapshots" => (*repo, *revision),
        _ => anyhow::bail!(
            "Invalid streamed Hugging Face snapshot path: {}",
            snapshot.display()
        ),
    };

    if let Some(expected) = resolved_revision {
        anyhow::ensure!(
            revision == expected,
            "Streamed Hugging Face snapshot {} does not match resolved revision {expected}",
            snapshot.display()
        );
    }

    let cache_root = cache.canonicalize()?;
    let repo = cache_child_dir(&cache_root, &cache_root, repo)?;
    let refs = cache_child_dir(&repo, &repo, OsStr::new("refs"))?;
    let snapshots = cache_child_dir(&repo, &repo, OsStr::new("snapshots"))?;
    let destination = snapshots.join(revision);
    let staged_ref = staging.join("main-ref");
    fs::write(&staged_ref, revision.as_encoded_bytes())?;

    // A cold snapshot becomes visible as a whole, including to readers that
    // enumerate snapshot directories instead of reading refs/main.
    match fs::rename(snapshot, &destination) {
        Ok(()) => {}
        Err(error)
            if matches!(
                error.kind(),
                ErrorKind::AlreadyExists | ErrorKind::DirectoryNotEmpty
            ) =>
        {
            let destination = cache_child_dir(&destination, &snapshots, revision)?;
            publish_snapshot_files(snapshot, &destination, &destination)?;
        }
        Err(error) => {
            return Err(error)
                .with_context(|| format!("Failed to install snapshot {}", destination.display()));
        }
    }
    // Update the ref only after the snapshot files have been published.
    let main_ref = refs.join("main");
    fs::rename(staged_ref, &main_ref)
        .with_context(|| format!("Failed to update Hugging Face ref {}", main_ref.display()))?;
    Ok(cache.join(relative))
}

fn publish_snapshot_files(
    source: &Path,
    destination: &Path,
    snapshot_root: &Path,
) -> anyhow::Result<()> {
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let name = entry.file_name();
        if entry.file_type()?.is_dir() {
            let child = cache_child_dir(snapshot_root, destination, &name)?;
            publish_snapshot_files(&entry.path(), &child, snapshot_root)?;
        } else {
            // Rename complete files over cache entries without following blob symlinks.
            let target = destination.join(name);
            fs::rename(entry.path(), &target)
                .with_context(|| format!("Failed to publish streamed file {}", target.display()))?;
        }
    }
    Ok(())
}

fn cache_child_dir(cache_root: &Path, parent: &Path, name: &OsStr) -> anyhow::Result<PathBuf> {
    let path = parent.join(name);
    match fs::create_dir(&path) {
        Ok(()) => Ok(path),
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {
            let canonical = path.canonicalize()?;
            anyhow::ensure!(
                canonical.is_dir() && canonical.starts_with(cache_root),
                "Invalid Hugging Face cache directory: {}",
                path.display()
            );
            Ok(canonical)
        }
        Err(error) => Err(error).with_context(|| format!("Failed to create {}", path.display())),
    }
}

async fn mx_download_direct(
    model_name: &str,
    provider: MxModelProvider,
    ignore_weights: bool,
) -> anyhow::Result<PathBuf> {
    let cache_dir = get_model_express_cache_dir(provider);
    mx::download_model(model_name, provider, Some(cache_dir), ignore_weights).await
}

// TODO: remove in the future. This is a temporary workaround to find common
// cache directory between client and server.
fn get_model_express_cache_dir(provider: MxModelProvider) -> PathBuf {
    if provider == MxModelProvider::Ngc {
        // ModelExpress 0.6 adds the provider's `ngc/` layout beneath this root.
        return mx_envs::cache_directory()
            .unwrap_or_else(|| mx_envs::home_dir_or_cwd().join(".cache"));
    }
    cache_dir_from_values(
        env::var(env_model::huggingface::HF_HUB_CACHE).ok(),
        env::var(env_model::huggingface::HF_HOME).ok(),
        env::var(env_model::model_express::MODEL_EXPRESS_CACHE_PATH).ok(),
        env::var("HOME").ok(),
        env::var("USERPROFILE").ok(),
    )
}

fn cache_dir_from_values(
    hf_hub_cache: Option<String>,
    hf_home: Option<String>,
    model_express_cache: Option<String>,
    home: Option<String>,
    userprofile: Option<String>,
) -> PathBuf {
    if let Some(cache_path) = hf_hub_cache {
        return PathBuf::from(cache_path);
    }
    if let Some(hf_home) = hf_home {
        return PathBuf::from(hf_home).join("hub");
    }
    if let Some(cache_path) = model_express_cache {
        return PathBuf::from(cache_path);
    }

    PathBuf::from(home.or(userprofile).unwrap_or_else(|| ".".to_string()))
        .join(".cache/huggingface/hub")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[serial_test::serial]
    #[test]
    fn hf_offline_mode_accepts_huggingface_truthy_values() {
        for value in ["1", "true", "TRUE", "on", "ON", "yes", "YES"] {
            temp_env::with_var(env_model::huggingface::HF_HUB_OFFLINE, Some(value), || {
                assert!(is_offline_mode(), "rejected {value}")
            });
        }
    }

    #[test]
    fn cache_dir_precedence_and_fallback() {
        assert_eq!(
            cache_dir_from_values(
                Some("/hub-cache".to_string()),
                Some("/hf-home".to_string()),
                Some("/model-express".to_string()),
                Some("/home".to_string()),
                None,
            ),
            PathBuf::from("/hub-cache")
        );
        assert_eq!(
            cache_dir_from_values(
                None,
                Some("/hf-home".to_string()),
                Some("/model-express".to_string()),
                Some("/home".to_string()),
                None,
            ),
            PathBuf::from("/hf-home/hub")
        );
        assert_eq!(
            cache_dir_from_values(
                None,
                None,
                Some("/model-express".to_string()),
                Some("/home".to_string()),
                None,
            ),
            PathBuf::from("/model-express")
        );
        assert_eq!(
            cache_dir_from_values(None, None, None, None, Some("/profile".to_string())),
            PathBuf::from("/profile/.cache/huggingface/hub")
        );
    }

    /// Build an hf-hub-format cache layout for `model_name` in `cache_root`,
    /// populated with the given filenames at a fake snapshot revision. Returns
    /// the snapshot directory path that `Cache::model().get()` should resolve to.
    fn build_hf_cache(cache_root: &Path, model_name: &str, files: &[&str]) -> PathBuf {
        let repo_dir = cache_root.join(format!("models--{}", model_name.replace('/', "--")));
        let snapshot_hash = "0000000000000000000000000000000000000000";
        let snapshot_dir = repo_dir.join("snapshots").join(snapshot_hash);
        let refs_dir = repo_dir.join("refs");
        fs::create_dir_all(&snapshot_dir).unwrap();
        fs::create_dir_all(&refs_dir).unwrap();
        fs::write(refs_dir.join("main"), snapshot_hash).unwrap();
        for f in files {
            fs::write(snapshot_dir.join(f), "{}").unwrap();
        }
        snapshot_dir
    }

    #[test]
    fn streamed_snapshot_publish_rejects_mismatched_revision() {
        let cache = TempDir::new().unwrap();
        let staging = TempDir::new_in(cache.path()).unwrap();
        let snapshot = build_hf_cache(staging.path(), "org/model", &["config.json"]);
        let error = publish_streamed_hf_snapshot(
            staging.path(),
            &snapshot,
            cache.path(),
            Some("1111111111111111111111111111111111111111"),
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("does not match resolved revision")
        );
        assert!(!cache.path().join("models--org--model").exists());
    }

    #[test]
    fn streamed_snapshot_publish_failure_leaves_previous_ref() {
        let cache = TempDir::new().unwrap();
        let staging = TempDir::new_in(cache.path()).unwrap();
        let model = "org/model";
        let files = ["config.json", "tokenizer.json"];
        let destination = build_hf_cache(cache.path(), model, &files);
        let refs = destination
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("refs/main");
        let previous = destination.with_file_name("1111111111111111111111111111111111111111");
        fs::rename(&destination, &previous).unwrap();
        fs::write(&refs, previous.file_name().unwrap().as_encoded_bytes()).unwrap();
        fs::create_dir_all(&destination).unwrap();
        fs::write(
            destination.join("nested"),
            b"cache entry blocks directory creation",
        )
        .unwrap();

        let snapshot = build_hf_cache(staging.path(), model, &files);
        fs::create_dir(snapshot.join("nested")).unwrap();
        fs::write(snapshot.join("nested/model.safetensors"), b"complete").unwrap();

        assert!(
            publish_streamed_hf_snapshot(staging.path(), &snapshot, cache.path(), None).is_err()
        );
        assert_eq!(
            get_cached_model_path_in(model, true, cache.path().to_path_buf()),
            Some(previous)
        );
    }

    #[cfg(unix)]
    #[test]
    fn streamed_snapshot_replaces_a_blob_symlink_without_changing_the_blob() {
        let cache = TempDir::new().unwrap();
        let staging = TempDir::new_in(cache.path()).unwrap();
        let model = "org/model";
        let destination = build_hf_cache(cache.path(), model, &["tokenizer.json"]);
        let blob = cache.path().join("blob");
        fs::write(&blob, b"prior blob").unwrap();
        std::os::unix::fs::symlink(&blob, destination.join("config.json")).unwrap();
        let snapshot = build_hf_cache(staging.path(), model, &["config.json", "model.safetensors"]);
        fs::write(snapshot.join("config.json"), b"new config").unwrap();
        fs::write(snapshot.join("model.safetensors"), b"complete").unwrap();

        assert_eq!(
            publish_streamed_hf_snapshot(staging.path(), &snapshot, cache.path(), None).unwrap(),
            destination
        );
        assert_eq!(fs::read(blob).unwrap(), b"prior blob");
        assert_eq!(
            fs::read(destination.join("config.json")).unwrap(),
            b"new config"
        );
        assert_eq!(
            fs::read(destination.join("model.safetensors")).unwrap(),
            b"complete"
        );
        assert_eq!(
            get_cached_model_path_in(model, false, cache.path().to_path_buf()),
            Some(destination)
        );
    }

    #[cfg(unix)]
    #[test]
    fn streamed_snapshot_publish_rejects_symlink_into_a_sibling_snapshot() {
        let cache = TempDir::new().unwrap();
        let staging = TempDir::new_in(cache.path()).unwrap();
        let model = "org/model";
        let destination = build_hf_cache(cache.path(), model, &["config.json", "tokenizer.json"]);
        let sibling = destination.with_file_name("1111111111111111111111111111111111111111");
        fs::create_dir(&sibling).unwrap();
        fs::write(sibling.join("config.json"), b"sibling").unwrap();
        std::os::unix::fs::symlink(&sibling, destination.join("nested")).unwrap();

        let snapshot = build_hf_cache(staging.path(), model, &["config.json"]);
        fs::create_dir(snapshot.join("nested")).unwrap();
        fs::write(snapshot.join("nested/config.json"), b"streamed").unwrap();

        assert!(
            publish_streamed_hf_snapshot(staging.path(), &snapshot, cache.path(), None).is_err()
        );
        assert_eq!(fs::read(sibling.join("config.json")).unwrap(), b"sibling");
    }

    #[test]
    fn test_cached_path_metadata_only_satisfies_ignore_weights_true() {
        // A cache with only metadata files should satisfy ignore_weights=true
        // but NOT ignore_weights=false (no weight files present).
        let temp = TempDir::new().unwrap();
        let model = "test-org/metadata-only";
        let snapshot = build_hf_cache(temp.path(), model, &["config.json", "tokenizer.json"]);

        let with_weights = get_cached_model_path_in(model, false, temp.path().to_path_buf());
        let no_weights = get_cached_model_path_in(model, true, temp.path().to_path_buf());

        assert!(
            with_weights.is_none(),
            "metadata-only cache must NOT satisfy ignore_weights=false"
        );
        assert_eq!(
            no_weights.as_deref(),
            Some(snapshot.as_path()),
            "metadata-only cache must satisfy ignore_weights=true"
        );
    }

    #[test]
    fn cache_lookup_rejects_directories_named_like_model_files() {
        let cache = TempDir::new().unwrap();
        let model = "org/model";
        let snapshot = build_hf_cache(cache.path(), model, &["config.json"]);
        fs::create_dir(snapshot.join("tokenizer.json")).unwrap();
        fs::create_dir(snapshot.join("tokenizer.tiktoken")).unwrap();
        assert!(get_cached_model_path_in(model, true, cache.path().to_path_buf()).is_none());

        fs::remove_dir(snapshot.join("tokenizer.json")).unwrap();
        fs::write(snapshot.join("tokenizer.json"), b"{}").unwrap();
        fs::create_dir(snapshot.join("model.safetensors")).unwrap();
        assert!(get_cached_model_path_in(model, false, cache.path().to_path_buf()).is_none());
    }

    #[test]
    fn test_cached_path_full_cache_satisfies_both_modes() {
        let temp = TempDir::new().unwrap();
        let model = "test-org/full-cache";
        let snapshot = build_hf_cache(
            temp.path(),
            model,
            &["config.json", "tokenizer.json", "model.safetensors"],
        );

        let with_weights = get_cached_model_path_in(model, false, temp.path().to_path_buf());
        let no_weights = get_cached_model_path_in(model, true, temp.path().to_path_buf());

        assert_eq!(with_weights.as_deref(), Some(snapshot.as_path()));
        assert_eq!(no_weights.as_deref(), Some(snapshot.as_path()));
    }

    #[test]
    fn test_cached_path_sharded_requires_all_shard_files() {
        // A cache containing only `model.safetensors.index.json` (without the
        // shard files it points to) is incomplete and must NOT satisfy
        // ignore_weights=false. Once all shards are written, it should.
        let temp = TempDir::new().unwrap();
        let model = "test-org/sharded";
        let snapshot = build_hf_cache(temp.path(), model, &["config.json", "tokenizer.json"]);
        fs::write(
            snapshot.join("model.safetensors.index.json"),
            r#"{"weight_map": {"a.weight": "model-00001-of-00002.safetensors", "b.weight": "model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();

        let incomplete = get_cached_model_path_in(model, false, temp.path().to_path_buf());
        assert!(
            incomplete.is_none(),
            "sharded cache without shard files must NOT satisfy ignore_weights=false"
        );

        fs::write(snapshot.join("model-00001-of-00002.safetensors"), "").unwrap();
        fs::write(snapshot.join("model-00002-of-00002.safetensors"), "").unwrap();
        let complete = get_cached_model_path_in(model, false, temp.path().to_path_buf());
        assert_eq!(complete.as_deref(), Some(snapshot.as_path()));
    }

    #[test]
    fn test_cached_path_rejects_tokenizer_config_without_real_tokenizer() {
        // A snapshot with only ``config.json`` and ``tokenizer_config.json``
        // (no ``tokenizer.json`` / ``tiktoken.model`` / ``*.tiktoken``) cannot
        // actually load a tokenizer at runtime via
        // ``TokenizerKind::from_disk``. The cache-hit probe must reject this
        // partial state in BOTH modes so ``from_hf`` falls through to a
        // download that populates the real tokenizer artifact.
        let temp = TempDir::new().unwrap();
        let model = "test-org/tokenizer-config-only";
        build_hf_cache(
            temp.path(),
            model,
            &["config.json", "tokenizer_config.json"],
        );

        assert!(
            get_cached_model_path_in(model, true, temp.path().to_path_buf()).is_none(),
            "tokenizer_config.json alone must NOT satisfy ignore_weights=true",
        );
        assert!(
            get_cached_model_path_in(model, false, temp.path().to_path_buf()).is_none(),
            "tokenizer_config.json alone must NOT satisfy ignore_weights=false",
        );
    }

    #[serial_test::serial]
    #[tokio::test]
    async fn test_from_hf_cache_first_in_online_mode() {
        // The cache-first short-circuit must fire even when HF_HUB_OFFLINE is
        // not set. If it does, from_hf returns the cached path without touching
        // MxClient or the HF network.
        let temp = TempDir::new().unwrap();
        let model = "test-org/cache-first-online";
        let snapshot = build_hf_cache(
            temp.path(),
            model,
            &["config.json", "tokenizer.json", "model.safetensors"],
        );

        temp_env::async_with_vars(
            [
                (
                    env_model::huggingface::HF_HUB_CACHE,
                    Some(temp.path().to_str().unwrap()),
                ),
                (env_model::huggingface::HF_HUB_OFFLINE, None),
                (env_model::huggingface::HF_HOME, None),
                (env_model::model_express::MODEL_EXPRESS_CACHE_PATH, None),
            ],
            async {
                let result = from_hf(PathBuf::from(model), false).await;

                assert_eq!(
                    result.ok().as_deref(),
                    Some(snapshot.as_path()),
                    "from_hf must return cached path in online mode without network"
                );
            },
        )
        .await;
    }

    #[test]
    fn provider_for_routes_ngc_uris_to_ngc() {
        assert_eq!(
            provider_for("ngc://nvstaging/nim/nemotron-3-ultra:hf-rl-052726-nvfp4-e9744cc"),
            MxModelProvider::Ngc
        );
        assert_eq!(
            provider_for("Qwen/Qwen3-0.6B"),
            MxModelProvider::HuggingFace
        );
        assert_eq!(
            provider_for("/data/llms/Qwen3-0.6B"),
            MxModelProvider::HuggingFace
        );
    }

    const NGC_TEST_MODEL: &str = "ngc://test-org/test-team/test-model:v1";

    fn build_ngc_cache(cache_root: &Path, files: &[&str]) -> PathBuf {
        let model_dir = cache_root.join("ngc/test-org/test-team/models/test-model/v1");
        fs::create_dir_all(&model_dir).unwrap();
        for f in files {
            fs::write(model_dir.join(f), "{}").unwrap();
        }
        model_dir
    }

    #[test]
    fn test_ngc_cached_path_requires_weights_for_full_download() {
        let temp = TempDir::new().unwrap();
        let model_dir = build_ngc_cache(temp.path(), &["config.json", "tokenizer.json"]);
        let lookup = |ignore_weights: bool| {
            get_cached_provider_model_path_in(
                NGC_TEST_MODEL,
                MxModelProvider::Ngc,
                ignore_weights,
                temp.path(),
            )
        };

        assert!(
            lookup(false).is_none(),
            "NGC cache without weights must NOT satisfy ignore_weights=false"
        );
        assert_eq!(lookup(true).as_deref(), Some(model_dir.as_path()));

        fs::write(model_dir.join("model.safetensors"), "").unwrap();
        assert_eq!(lookup(false).as_deref(), Some(model_dir.as_path()));
    }

    #[serial_test::serial]
    #[tokio::test]
    async fn test_from_hf_ngc_cache_first() {
        for explicit_cache in [true, false] {
            let temp = TempDir::new().unwrap();
            let home = temp.path().join("home");
            let hf_cache = temp.path().join("hf");
            let cache_root = if explicit_cache {
                temp.path().join("mx")
            } else {
                home.join(".cache")
            };
            let model_dir = build_ngc_cache(
                &cache_root,
                &["config.json", "tokenizer.json", "model.safetensors"],
            );

            temp_env::async_with_vars(
                [
                    (
                        mx_envs::MODEL_EXPRESS_CACHE_DIRECTORY,
                        explicit_cache.then(|| cache_root.to_str().unwrap()),
                    ),
                    (mx_envs::HOME, home.to_str()),
                    (env_model::huggingface::HF_HUB_CACHE, hf_cache.to_str()),
                    (env_model::huggingface::HF_HOME, hf_cache.to_str()),
                    (
                        env_model::model_express::MODEL_EXPRESS_CACHE_PATH,
                        hf_cache.to_str(),
                    ),
                ],
                async {
                    assert_eq!(
                        get_cached_model_path(NGC_TEST_MODEL, MxModelProvider::Ngc, false)
                            .as_deref(),
                        Some(model_dir.as_path())
                    );
                    assert_eq!(
                        get_model_express_cache_dir(MxModelProvider::HuggingFace),
                        hf_cache
                    );
                    let result = from_hf(PathBuf::from(NGC_TEST_MODEL), false).await;
                    assert_eq!(result.ok().as_deref(), Some(model_dir.as_path()));
                },
            )
            .await;
        }
    }
}
