// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::OsStr;
use std::io::{self, ErrorKind};
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
/// For metadata-only downloads (ignore_weights=true), check the configuration and
/// tokenizer for the model format. Full downloads also require weights.
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
    let repo = cache.model(model_name.to_string());
    let snapshot_path = repo
        .get("config.json")
        .or_else(|| repo.get("params.json"))?
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
    if !has_required_model_files(&dir, ignore_weights).unwrap_or(false) {
        return None;
    }
    tracing::info!("Found cached model '{model_name}' at {dir:?}, skipping download");
    Some(dir)
}

fn cached_file(path: &Path) -> io::Result<bool> {
    match fs::metadata(path) {
        Ok(metadata) => Ok(metadata.is_file()),
        Err(error) if matches!(error.kind(), ErrorKind::NotFound | ErrorKind::NotADirectory) => {
            Ok(false)
        }
        Err(error) => Err(error),
    }
}

/// Require a configuration and tokenizer in the same model format, plus weights
/// unless explicitly skipped. Preserve I/O errors for callers updating refs.
fn has_required_model_files(dir: &Path, ignore_weights: bool) -> io::Result<bool> {
    let has = |filename: &str| cached_file(&dir.join(filename));
    let metadata = match fs::metadata(dir.join("config.json")) {
        Ok(config) => {
            config.is_file()
                && (has("tokenizer.json")? || has("tiktoken.model")? || has_tiktoken_file(dir)?)
        }
        Err(error) if error.kind() == ErrorKind::NotFound => {
            has("params.json")? && has("tekken.json")?
        }
        Err(error) => return Err(error),
    };
    Ok(metadata
        && (ignore_weights
            || has("model.safetensors")?
            || has("pytorch_model.bin")?
            || shard_files_present(&dir.join("model.safetensors.index.json"))?
            || shard_files_present(&dir.join("pytorch_model.bin.index.json"))?))
}

fn has_tiktoken_file(dir: &Path) -> io::Result<bool> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|ext| ext == "tiktoken") && cached_file(&path)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// An index counts as weights only when every referenced shard is present.
fn shard_files_present(index_path: &Path) -> io::Result<bool> {
    let Some(snapshot_dir) = index_path.parent() else {
        return Ok(false);
    };
    if !cached_file(index_path)? {
        return Ok(false);
    }
    let contents = fs::read(index_path)?;
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&contents) else {
        return Ok(false);
    };
    let Some(weight_map) = value.get("weight_map").and_then(|v| v.as_object()) else {
        return Ok(false);
    };
    let mut checked = std::collections::HashSet::new();
    for shard in weight_map.values() {
        let Some(name) = shard.as_str() else {
            return Ok(false);
        };
        if checked.insert(name) && !cached_file(&snapshot_dir.join(name))? {
            return Ok(false);
        }
    }
    Ok(!checked.is_empty())
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
    fetch_snapshot(name.as_ref(), SnapshotRequest::Model { ignore_weights }).await
}

enum SnapshotRequest {
    Model { ignore_weights: bool },
    CheckedMetadata,
}

pub(super) async fn snapshot_for_checked_metadata(repo: &str) -> anyhow::Result<PathBuf> {
    fetch_snapshot(Path::new(repo), SnapshotRequest::CheckedMetadata).await
}

async fn fetch_snapshot(name: &Path, request: SnapshotRequest) -> anyhow::Result<PathBuf> {
    let ignore_weights = match request {
        SnapshotRequest::Model { ignore_weights } => ignore_weights,
        SnapshotRequest::CheckedMetadata => true,
    };
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
        // Both downloaders publish into the same HF cache.
        Err(error) if error.is::<InvalidCacheDirectory>() => Err(error),
        Err(server_error) => {
            tracing::warn!(%model_name, error = %server_error, "ModelExpress fetch failed, falling back to direct download");
            mx_download_direct(&model_name, provider, ignore_weights)
                .await
                .and_then(|path| {
                    // The offline provider can resolve a snapshot without checking its files.
                    // Model cards verify each advertised metadata file against its checksum.
                    if hf_stream_cache.is_some() && mx_envs::hf_offline()
                        && matches!(request, SnapshotRequest::Model { .. }) {
                        anyhow::ensure!(
                            has_required_model_files(&path, ignore_weights)
                                .with_context(|| format!("Failed to inspect cached snapshot {}", path.display()))?,
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
        let cache = cache.to_path_buf();
        let (staging, cache) = tokio::task::spawn_blocking(move || {
            let staging = StreamStaging::new(&cache)
                .with_context(|| format!("Failed to stage a download in {}", cache.display()))?;
            anyhow::Ok((staging, cache))
        })
        .await??;
        config.cache.local_path = staging.path().to_path_buf();
        Some((staging, cache))
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
                ignore_weights,
            )
        })
        .await?;
    }

    client.get_model_path(model_name, provider).await
}

struct StreamStaging {
    // Drop the directory while we still own its slot.
    dir: tempfile::TempDir,
    _owner: fs::File,
}

impl StreamStaging {
    fn new(cache: &Path) -> std::io::Result<Self> {
        // HF cache tools skip .locks when enumerating model repositories.
        let locks = cache.join(".locks");
        fs::create_dir_all(&locks)?;
        if !fs::symlink_metadata(&locks)?.is_dir() {
            return Err(std::io::Error::other(
                "Invalid Hugging Face .locks directory",
            ));
        }
        let mut reserved = None;
        let mut index = 0_u64;
        let (slot, owner) = loop {
            // Keep each lock inode across transfers, including while its data is removed.
            let name = format!("dynamo-mx-stream-{index}");
            index += 1;
            let lock_path = locks.join(format!("{name}.lock"));
            let owner = match lock_stream_slot(&lock_path, reserved.is_none()) {
                Ok(Some(owner)) => owner,
                Ok(None) => continue,
                // Once we have a slot, only visit slots that already exist.
                Err(error) if error.kind() == ErrorKind::NotFound => {
                    break reserved.ok_or(error)?;
                }
                Err(error) => {
                    if let Some(slot) = reserved {
                        tracing::warn!(path = %lock_path.display(), %error, "Could not check remaining ModelExpress staging slots");
                        break slot;
                    }
                    return Err(error);
                }
            };

            let path = locks.join(&name);
            match fs::remove_dir_all(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == ErrorKind::NotFound => {}
                Err(error)
                    if reserved.is_some()
                        || matches!(
                            error.kind(),
                            ErrorKind::PermissionDenied
                                | ErrorKind::DirectoryNotEmpty
                                | ErrorKind::NotADirectory
                        ) =>
                {
                    tracing::warn!(path = %path.display(), %error, "Failed to reclaim ModelExpress staging directory");
                    continue;
                }
                Err(error) => return Err(error),
            }
            if reserved.is_none() {
                reserved = Some((path, owner));
            }
        };
        fs::create_dir(&slot)?;
        // A cancelled async filesystem operation may still finish on a blocking
        // thread. Give each transfer its own path, even when reusing a slot.
        let dir = tempfile::tempdir_in(slot)?;
        Ok(Self { dir, _owner: owner })
    }

    fn path(&self) -> &Path {
        self.dir.path()
    }
}

fn open_cache_lock(path: &Path, create: bool) -> std::io::Result<fs::File> {
    let mut options = fs::File::options();
    options
        .read(true)
        .write(true)
        .create(create)
        .truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err(std::io::Error::other(
            "Invalid ModelExpress cache lock file",
        ));
    }
    Ok(file)
}

fn lock_stream_slot(path: &Path, create: bool) -> std::io::Result<Option<fs::File>> {
    let owner = match open_cache_lock(path, create) {
        Ok(file) => file,
        // Skip an unusable entry. If the directory itself is unwritable,
        // creating the first absent lock must still fail.
        Err(error)
            if fs::symlink_metadata(path).is_ok_and(|entry| {
                !entry.is_file() || error.kind() == ErrorKind::PermissionDenied
            }) =>
        {
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    match owner.try_lock() {
        Ok(()) => Ok(Some(owner)),
        Err(fs::TryLockError::WouldBlock) => Ok(None),
        Err(fs::TryLockError::Error(error)) => Err(error),
    }
}

fn publish_streamed_hf_snapshot(
    staging: &Path,
    snapshot: &Path,
    cache: &Path,
    resolved_revision: Option<&str>,
    ignore_weights: bool,
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
    let repo_dir = cache_child_dir(&cache_root, repo)?;
    let refs = cache_child_dir(&repo_dir, OsStr::new("refs"))?;
    let snapshots = cache_child_dir(&repo_dir, OsStr::new("snapshots"))?;
    let locks = cache_child_dir(&cache_root, OsStr::new(".locks"))?;
    let repo_hash = blake3::hash(repo.as_encoded_bytes());
    let lock_path = locks.join(format!("dynamo-mx-main-{repo_hash}.lock"));
    // Serialize stream publication per repo, including merges that can make
    // the currently referenced snapshot complete.
    let lock = open_cache_lock(&lock_path, true)
        .with_context(|| format!("Failed to open publication lock {}", lock_path.display()))?;
    lock.lock()
        .with_context(|| format!("Failed to lock {}", lock_path.display()))?;
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
            let destination = cache_child_dir(&snapshots, revision)?;
            publish_snapshot_files(snapshot, &destination)?;
        }
        Err(error)
            if error.kind() == ErrorKind::NotADirectory
                && fs::symlink_metadata(&destination).is_ok_and(|entry| !entry.is_dir()) =>
        {
            return Err(InvalidCacheDirectory(destination).into());
        }
        Err(error) => {
            return Err(error)
                .with_context(|| format!("Failed to install snapshot {}", destination.display()));
        }
    }
    // Update the ref only after the snapshot files have been published.
    let main_ref = refs.join("main");
    let preserve_main = ignore_weights && main_has_complete_snapshot(&main_ref, &snapshots)
        .unwrap_or_else(|error| {
            tracing::warn!(path = %main_ref.display(), %error, "Preserving the current ref because snapshot validation failed");
            true
        });
    if !preserve_main {
        fs::rename(staged_ref, &main_ref)
            .with_context(|| format!("Failed to update Hugging Face ref {}", main_ref.display()))?;
    }
    Ok(cache.join(relative))
}

fn main_has_complete_snapshot(main_ref: &Path, snapshots: &Path) -> io::Result<bool> {
    let revision = match fs::read_to_string(main_ref) {
        Ok(revision) => revision,
        Err(error) if matches!(error.kind(), ErrorKind::NotFound | ErrorKind::InvalidData) => {
            return Ok(false);
        }
        Err(error) => return Err(error),
    };
    let revision = Path::new(&revision);
    if revision.file_name() != Some(revision.as_os_str()) {
        return Ok(false);
    }
    has_required_model_files(&snapshots.join(revision), false)
}

fn publish_snapshot_files(source: &Path, destination: &Path) -> anyhow::Result<()> {
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let name = entry.file_name();
        if entry.file_type()?.is_dir() {
            let child = cache_child_dir(destination, &name)?;
            publish_snapshot_files(&entry.path(), &child)?;
        } else {
            // Rename complete files over cache entries without following blob symlinks.
            let target = destination.join(name);
            fs::rename(entry.path(), &target)
                .with_context(|| format!("Failed to publish streamed file {}", target.display()))?;
        }
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
#[error("Invalid Hugging Face cache directory: {}", .0.display())]
struct InvalidCacheDirectory(PathBuf);

fn cache_child_dir(parent: &Path, name: &OsStr) -> anyhow::Result<PathBuf> {
    let path = parent.join(name);
    match fs::create_dir(&path) {
        Ok(()) => Ok(path),
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {
            if !fs::symlink_metadata(&path)?.is_dir() {
                return Err(InvalidCacheDirectory(path).into());
            }
            Ok(path)
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
    fn stream_staging_reclaims_abandoned_slots_and_preserves_live_downloads() {
        let cache = TempDir::new().unwrap();
        let active = StreamStaging::new(cache.path()).unwrap();
        let abandoned = StreamStaging::new(cache.path()).unwrap();
        let higher_slot = StreamStaging::new(cache.path()).unwrap();
        fs::write(active.path().join("partial"), b"live").unwrap();

        // Simulate the filesystem state after the kernel releases a dead owner's lock.
        let abandon = |stage: StreamStaging| {
            let path = stage.dir.keep();
            fs::write(path.join("partial"), b"abandoned").unwrap();
            path
        };
        let stale = abandon(abandoned);
        let higher = abandon(higher_slot);
        let retry = StreamStaging::new(cache.path()).unwrap();
        assert_eq!(retry.path().parent(), stale.parent());
        assert_ne!(retry.path(), stale);
        assert!(!stale.exists());
        assert!(!higher.exists());
        assert_eq!(fs::read(active.path().join("partial")).unwrap(), b"live");
    }

    #[test]
    fn stream_staging_skips_an_unusable_slot() {
        let cache = TempDir::new().unwrap();
        let first = StreamStaging::new(cache.path()).unwrap();
        let slot = first.path().parent().unwrap().to_owned();
        drop(first);
        fs::remove_dir(&slot).unwrap();
        fs::write(&slot, b"keep").unwrap();
        let retry = StreamStaging::new(cache.path()).unwrap();
        assert_ne!(retry.path().parent(), Some(slot.as_path()));
        assert_eq!(fs::read(slot).unwrap(), b"keep");
    }

    #[test]
    fn stream_staging_skips_a_directory_at_the_lock_path() {
        let cache = TempDir::new().unwrap();
        let locks = cache.path().join(".locks");
        let blocked = locks.join("dynamo-mx-stream-0.lock");
        fs::create_dir_all(&blocked).unwrap();

        let staging = StreamStaging::new(cache.path()).unwrap();
        assert!(staging.path().starts_with(locks.join("dynamo-mx-stream-1")));
        assert!(blocked.is_dir());
    }

    #[cfg(unix)]
    #[test]
    fn cache_child_rejects_an_alias_within_its_parent() {
        let cache = TempDir::new().unwrap();
        let parent = cache.path().canonicalize().unwrap();
        let actual = parent.join("models--org--actual");
        fs::create_dir(&actual).unwrap();
        std::os::unix::fs::symlink(&actual, parent.join("models--org--alias")).unwrap();

        assert!(cache_child_dir(&parent, OsStr::new("models--org--alias")).is_err());
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
            false,
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
    fn metadata_publication_preserves_a_complete_main_revision() {
        let cache = TempDir::new().unwrap();
        let model = "org/model";
        let publish = |revision: &str, ignore_weights: bool| {
            let staging = TempDir::new_in(cache.path()).unwrap();
            let original =
                build_hf_cache(staging.path(), model, &["config.json", "tokenizer.json"]);
            let snapshot = original.with_file_name(revision);
            fs::rename(original, &snapshot).unwrap();
            if !ignore_weights {
                fs::write(snapshot.join("model.safetensors"), b"weights").unwrap();
            }
            publish_streamed_hf_snapshot(
                staging.path(),
                &snapshot,
                cache.path(),
                Some(revision),
                ignore_weights,
            )
            .unwrap()
        };
        let full = publish("full-rev", false);
        let metadata = publish("metadata-rev", true);
        assert!(metadata.join("config.json").is_file());
        assert_eq!(
            get_cached_model_path_in(model, false, cache.path().into()),
            Some(full.clone())
        );

        fs::remove_file(full.join("model.safetensors")).unwrap();
        publish("metadata-rev", true);
        assert_eq!(
            get_cached_model_path_in(model, true, cache.path().into()),
            Some(metadata)
        );
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
            publish_streamed_hf_snapshot(staging.path(), &snapshot, cache.path(), None, false)
                .is_err()
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
            publish_streamed_hf_snapshot(staging.path(), &snapshot, cache.path(), None, false)
                .unwrap(),
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
            publish_streamed_hf_snapshot(staging.path(), &snapshot, cache.path(), None, false)
                .is_err()
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
    fn native_metadata_cache_hit_still_requires_weights_for_full_fetch() {
        let cache = TempDir::new().unwrap();
        let model = "org/native";
        let snapshot = build_hf_cache(cache.path(), model, &["params.json", "tekken.json"]);
        assert_eq!(
            get_cached_model_path_in(model, true, cache.path().into()),
            Some(snapshot.clone())
        );
        assert!(get_cached_model_path_in(model, false, cache.path().into()).is_none());
        fs::write(snapshot.join("config.json"), "{}").unwrap();
        assert!(get_cached_model_path_in(model, true, cache.path().into()).is_none());
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
