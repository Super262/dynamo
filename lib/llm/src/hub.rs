// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::OsStr;
use std::io::{self, ErrorKind};
use std::path::{Component, Path, PathBuf};
use std::{env, fs};

use crate::common::checked_file::CheckedFile;
use anyhow::Context;
use hf_hub::Cache;
use modelexpress_client::{
    Client as MxClient, ClientConfig as MxClientConfig, ModelProvider as MxModelProvider,
};
use modelexpress_common::{cache as mx_cache, download as mx, envs as mx_envs};
use std::sync::Arc;

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
) -> anyhow::Result<Option<PathBuf>> {
    let cache_dir = get_model_express_cache_dir(provider);
    match provider {
        MxModelProvider::HuggingFace => {
            get_cached_model_path_in(model_name, ignore_weights, cache_dir)
        }
        _ => Ok(get_cached_provider_model_path_in(
            model_name,
            provider,
            ignore_weights,
            &cache_dir,
        )),
    }
}

fn get_cached_model_path_in(
    model_name: &str,
    ignore_weights: bool,
    cache_dir: PathBuf,
) -> anyhow::Result<Option<PathBuf>> {
    let repo_dir = cache_dir.join(hf_hub::Repo::model(model_name.to_owned()).folder_name());
    let repo = Cache::new(cache_dir).model(model_name.to_owned());
    read_hf_checkout(&repo_dir, || {
        let snapshot = repo.get("config.json").or_else(|| repo.get("params.json"));
        Ok(snapshot.and_then(|path| {
            cached_model_dir(model_name, path.parent()?.to_owned(), ignore_weights)
        }))
    })
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
    let native_metadata = || Ok(has("params.json")? && has("tekken.json")?);
    let hf_metadata = || {
        let tokenizers = ["tokenizer.json", "tiktoken.model"].into_iter().map(has);
        Ok(has("config.json")?
            && any_available(tokenizers.chain(std::iter::once_with(|| has_tiktoken_file(dir))))?)
    };
    if !any_available(
        std::iter::once_with(hf_metadata).chain(std::iter::once_with(native_metadata)),
    )? {
        return Ok(false);
    }
    if ignore_weights {
        return Ok(true);
    }

    let weights = ["model.safetensors", "pytorch_model.bin"]
        .into_iter()
        .map(has)
        .chain(std::iter::once_with(|| {
            Ok(has("consolidated.safetensors")? && native_metadata()?)
        }));
    let indexes = [
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ]
    .into_iter()
    .map(|name| shard_files_present(&dir.join(name)));
    any_available(weights.chain(indexes))
}

// An unreadable alternative matters only if none of the others is usable.
fn any_available(checks: impl Iterator<Item = io::Result<bool>>) -> io::Result<bool> {
    let mut first_error = None;
    for check in checks {
        match check {
            Ok(true) => return Ok(true),
            Ok(false) => {}
            Err(error) => {
                first_error.get_or_insert(error);
            }
        }
    }
    first_error.map_or(Ok(false), Err)
}

fn has_tiktoken_file(dir: &Path) -> io::Result<bool> {
    any_available(fs::read_dir(dir)?.map(|entry| {
        let path = entry?.path();
        if path.extension().is_some_and(|ext| ext == "tiktoken") {
            cached_file(&path)
        } else {
            Ok(false)
        }
    }))
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
        let shard = Path::new(name);
        if shard.as_os_str().is_empty()
            || shard
                .components()
                .any(|c| !matches!(c, Component::Normal(_) | Component::CurDir))
        {
            return Ok(false);
        }
        if checked.insert(name) {
            let path = snapshot_dir.join(shard);
            if !cached_file(&path)? || same_file::is_same_file(index_path, &path)? {
                return Ok(false);
            }
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

#[derive(Clone)]
enum SnapshotRequest {
    Model { ignore_weights: bool },
    CheckedMetadata(Arc<[(String, CheckedFile)]>),
}

impl SnapshotRequest {
    fn ignore_weights(&self) -> bool {
        match self {
            Self::Model { ignore_weights } => *ignore_weights,
            Self::CheckedMetadata(_) => true,
        }
    }
}

pub(super) fn hf_repo_cache_dir(repo: &str) -> PathBuf {
    get_model_express_cache_dir(MxModelProvider::HuggingFace)
        .join(hf_hub::Repo::model(repo.to_owned()).folder_name())
}

pub(super) async fn snapshot_for_checked_metadata(
    repo: &str,
    files: Arc<[(String, CheckedFile)]>,
) -> anyhow::Result<PathBuf> {
    fetch_snapshot(Path::new(repo), SnapshotRequest::CheckedMetadata(files)).await
}

async fn fetch_snapshot(name: &Path, request: SnapshotRequest) -> anyhow::Result<PathBuf> {
    let ignore_weights = request.ignore_weights();
    let model_name = name.display().to_string();
    let provider = provider_for(&model_name);

    // Cache-first in all modes: if the snapshot is already on disk with the files we
    // need, return it without touching the network.
    if matches!(request, SnapshotRequest::Model { .. }) {
        let model = model_name.clone();
        if let Some(path) = tokio::task::spawn_blocking(move || {
            get_cached_model_path(&model, provider, ignore_weights)
        })
        .await??
        {
            return Ok(path);
        }
    }

    let offline = is_offline_mode();
    if offline {
        tracing::warn!(
            "Offline mode enabled but model '{model_name}' not found in cache, attempting download anyway"
        );
    }

    let actual_provider = MxModelProvider::resolve_provider_for_model_name(&model_name, provider);
    let config = mx_client_config(actual_provider);
    let is_hf = actual_provider == MxModelProvider::HuggingFace;
    let hf_stream_cache =
        (!config.cache.shared_storage && is_hf).then(|| get_model_express_cache_dir(provider));
    let result = match mx_download_from_server(
        &model_name,
        provider,
        request.clone(),
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
            mx_download_direct(&model_name, provider, request, hf_stream_cache.is_some(), is_hf).await
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
    if matches!(
        provider,
        MxModelProvider::Ngc | MxModelProvider::HuggingFace
    ) {
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
    request: SnapshotRequest,
    mut config: MxClientConfig,
    hf_stream_cache: Option<&Path>,
) -> anyhow::Result<PathBuf> {
    let staging = if let Some(cache) = hf_stream_cache {
        let (staging, cache) = create_staging(cache).await?;
        config.cache.local_path = staging.path().to_path_buf();
        Some((staging, cache))
    } else {
        None
    };

    let mut client = MxClient::new(config).await?;
    tracing::info!("Successfully connected to ModelExpress server");
    let download = client
        .request_model_revision(model_name, provider, request.ignore_weights(), None)
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
                request,
            )
        })
        .await?;
    }

    // In shared storage the server already resolved the immutable revision.
    // Use that checkout even if this client cannot contact the upstream Hub.
    let is_hf = MxModelProvider::resolve_provider_for_model_name(model_name, provider)
        == MxModelProvider::HuggingFace;
    if is_hf
        && let Some(path) = download.path
        && let Some(rev) = &download.resolved_revision
        && Path::new(rev).file_name() == Some(OsStr::new(rev))
        && path.is_dir()
    {
        return local_hf_snapshot(model_name, path, None).await;
    }
    let path = client.get_model_path(model_name, provider).await?;
    if is_hf {
        local_hf_snapshot(model_name, path, None).await
    } else {
        Ok(path)
    }
}

// Shared-storage models may use provider-specific layouts. All local HF paths
// still wait for an in-flight publisher or recover an interrupted merge.
async fn local_hf_snapshot(
    model: &str,
    path: PathBuf,
    ignore_weights: Option<bool>,
) -> anyhow::Result<PathBuf> {
    let repo = hf_repo_cache_dir(model);
    tokio::task::spawn_blocking(move || {
        read_hf_checkout(&repo, || {
            anyhow::ensure!(
                path.is_dir(),
                "Missing Hugging Face snapshot at {}",
                path.display()
            );
            if let Some(ignore) = ignore_weights {
                anyhow::ensure!(
                    has_required_model_files(&path, ignore).with_context(|| format!(
                        "Failed to inspect cached snapshot {}",
                        path.display()
                    ))?,
                    "Cached Hugging Face snapshot at {} is incomplete",
                    path.display()
                );
            }
            Ok(())
        })?;
        Ok(path)
    })
    .await?
}

async fn create_staging(cache: &Path) -> anyhow::Result<(StreamStaging, PathBuf)> {
    let cache = cache.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let staging = StreamStaging::new(&cache)
            .with_context(|| format!("Failed to stage a download in {}", cache.display()))?;
        Ok((staging, cache))
    })
    .await?
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
    request: SnapshotRequest,
) -> anyhow::Result<PathBuf> {
    publish_hf_snapshot(staging, snapshot, cache, resolved_revision, request, true)
}

fn publish_hf_snapshot(
    staging: &Path,
    snapshot: &Path,
    cache: &Path,
    resolved_revision: Option<&str>,
    request: SnapshotRequest,
    validate_model: bool,
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

    match &request {
        SnapshotRequest::Model { ignore_weights } if validate_model => {
            anyhow::ensure!(
                has_required_model_files(snapshot, *ignore_weights).with_context(|| format!(
                    "Failed to inspect staged snapshot {}",
                    snapshot.display()
                ))?,
                "Staged Hugging Face snapshot at {} is incomplete",
                snapshot.display()
            );
        }
        // Shared-storage direct callers also support provider-specific model layouts.
        SnapshotRequest::Model { .. } => {}
        SnapshotRequest::CheckedMetadata(files) => {
            crate::model_card::matching_metadata_files(snapshot, files, None)
                .context("Downloaded metadata does not match the model card")?;
        }
    }

    let cache_root = cache.canonicalize()?;
    let repo_dir = cache_child_dir(&cache_root, repo)?;
    let refs = cache_child_dir(&repo_dir, OsStr::new("refs"))?;
    let snapshots = cache_child_dir(&repo_dir, OsStr::new("snapshots"))?;
    let _lock = lock_hf_publication(&cache_root, repo)?;
    let main_ref = refs.join("main");
    // hf-hub's pinned downloads also create refs/<commit>. Its offline reader
    // uses that ref when resolving files by commit instead of by main.
    let pinned_ref = staging.join(repo).join("refs").join(revision);
    let publish_pinned = revision != "main" && cached_file(&pinned_ref)?;
    if publish_pinned {
        anyhow::ensure!(
            fs::read(&pinned_ref)? == revision.as_encoded_bytes(),
            "Staged Hugging Face commit ref does not match its snapshot"
        );
    }
    let pinned_target = refs.join(revision);
    for target in std::iter::once(&main_ref).chain(publish_pinned.then_some(&pinned_target)) {
        match fs::symlink_metadata(target) {
            Ok(meta) if meta.is_dir() => return Err(InvalidCacheDirectory(target.clone()).into()),
            Err(error) if error.kind() != ErrorKind::NotFound => return Err(error.into()),
            _ => {}
        }
    }
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
            let mut moves = Vec::new();
            // A checked card only vouches for its advertised files. Add missing
            // siblings, but do not overwrite an existing sibling with unchecked bytes.
            let may_replace = |path: &Path| match &request {
                SnapshotRequest::CheckedMetadata(files) => {
                    files.iter().any(|(name, _)| path == snapshot.join(name))
                }
                SnapshotRequest::Model { .. } => true,
            };
            plan_snapshot_merge(snapshot, &destination, &may_replace, &mut moves)?;
            let journal =
                hf_publication_lock_path(&cache_root.join(".locks"), repo).with_extension("merge");
            install_snapshot_merge(staging, &repo_dir, &journal, &moves)?;
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
    let preserve_main = request.ignore_weights() && main_has_complete_snapshot(&main_ref, &snapshots)
        .unwrap_or_else(|error| {
            tracing::warn!(path = %main_ref.display(), %error, "Preserving the current ref because snapshot validation failed");
            true
        });
    if publish_pinned {
        fs::rename(pinned_ref, &pinned_target).with_context(|| {
            format!(
                "Failed to update pinned Hugging Face ref {}",
                pinned_target.display()
            )
        })?;
    }
    if !preserve_main {
        fs::rename(staged_ref, &main_ref)
            .with_context(|| format!("Failed to update Hugging Face ref {}", main_ref.display()))?;
    }
    Ok(cache.join(relative))
}

// Both the streamed and direct download paths use this publication lock.
fn lock_hf_publication(cache_root: &Path, repo: &OsStr) -> anyhow::Result<fs::File> {
    let locks = cache_child_dir(cache_root, OsStr::new(".locks"))?;
    let lock_path = hf_publication_lock_path(&locks, repo);
    let lock = open_cache_lock(&lock_path, true)
        .with_context(|| format!("Failed to open publication lock {}", lock_path.display()))?;
    lock.lock()
        .with_context(|| format!("Failed to lock {}", lock_path.display()))?;
    let journal = lock_path.with_extension("merge");
    if entry_exists(&journal)? {
        let cleanup = StreamStaging::new(cache_root)?;
        rollback_snapshot_merge(
            &cache_root.join(repo),
            &journal,
            &cleanup.path().join("recovered"),
        )
        .with_context(|| {
            format!(
                "Recovering interrupted HF snapshot merge {}",
                journal.display()
            )
        })?;
    }
    Ok(lock)
}

fn hf_publication_lock_path(locks: &Path, repo: &OsStr) -> PathBuf {
    let hash = blake3::hash(repo.as_encoded_bytes());
    locks.join(format!("dynamo-mx-main-{hash}.lock"))
}

// Native HF caches may be read-only and have no Dynamo lock yet. Writers
// create the persistent lock before publishing. If the first writer arrives
// during an optimistic read, discard that capture and repeat under its lock.
pub(super) fn read_hf_checkout<T>(
    repo: &Path,
    read: impl Fn() -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    let cache = repo.parent().context("HF repository has no cache root")?;
    let name = repo.file_name().context("HF repository has no name")?;
    let lock_path = hf_publication_lock_path(&cache.join(".locks"), name);
    let open = || -> anyhow::Result<Option<fs::File>> {
        let mut options = fs::File::options();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        }
        match options.open(&lock_path) {
            Ok(lock) => {
                if !lock.metadata()?.is_file() {
                    return Err(io::Error::other("Invalid HF publication lock").into());
                }
                lock.lock_shared()?;
                if entry_exists(&lock_path.with_extension("merge"))? {
                    drop(lock);
                    return Ok(Some(lock_hf_publication(cache, name)?));
                }
                Ok(Some(lock))
            }
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    };
    if let Some(_guard) = open()? {
        return read();
    }
    let captured = read();
    if let Some(_guard) = open()? {
        return read();
    }
    captured
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

// Keep backups outside the staging slot while a merge is live. A later Dynamo
// reader or publisher can undo a worker killed between two leaf renames.
fn install_snapshot_merge(
    staging: &Path,
    repo: &Path,
    journal: &Path,
    moves: &[(PathBuf, PathBuf)],
) -> anyhow::Result<()> {
    if let [(source, target)] = moves {
        fs::rename(source, target)?;
        return Ok(());
    }
    if moves.is_empty() {
        return Ok(());
    }
    let prepared = staging.join("merge");
    let mut dir = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        dir.mode(0o700);
    }
    dir.create(&prepared)?;
    let mut entries = Vec::with_capacity(moves.len());
    for (i, (source, target)) in moves.iter().enumerate() {
        let existed = entry_exists(target)?;
        entries.push((target.strip_prefix(repo)?.to_owned(), existed));
        fs::rename(source, prepared.join(i.to_string()))?;
    }
    fs::write(prepared.join("manifest"), serde_json::to_vec(&entries)?)?;
    fs::rename(&prepared, journal)?;
    let install = (|| -> io::Result<()> {
        for (i, (relative, existed)) in entries.iter().enumerate() {
            let target = repo.join(relative);
            if *existed {
                let old = journal.join(format!("{i}.old"));
                // Keep the name readable across replacement when hard links are
                // supported. Renaming also works on protected multi-user caches.
                if fs::hard_link(&target, &old).is_err() {
                    fs::rename(&target, old)?;
                }
            }
            fs::rename(journal.join(i.to_string()), target)?;
        }
        // This rename commits the entire merge and transfers cleanup to staging.
        fs::rename(journal, &prepared)
    })();
    if let Err(error) = install {
        rollback_snapshot_merge(repo, journal, &prepared)
            .with_context(|| format!("Recovering HF checkout after publication failed: {error}"))?;
        return Err(error).context("Publishing HF snapshot files");
    }
    Ok(())
}

fn entry_exists(path: &Path) -> io::Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

fn rollback_snapshot_merge(repo: &Path, journal: &Path, retired: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        fs::symlink_metadata(journal)?.is_dir(),
        "Invalid HF merge journal"
    );
    let entries: Vec<(PathBuf, bool)> =
        serde_json::from_slice(&fs::read(journal.join("manifest"))?)?;
    for (i, (relative, existed)) in entries.iter().enumerate().rev() {
        let components: Vec<_> = relative.components().collect();
        anyhow::ensure!(
            matches!(components.as_slice(), [Component::Normal(prefix), Component::Normal(_), ..] if *prefix == "snapshots")
                && components.len() >= 3
                && components.iter().all(|c| matches!(c, Component::Normal(_))),
            "Invalid path in HF merge journal"
        );
        let mut target = repo.to_owned();
        for component in &components[..components.len() - 1] {
            target.push(component);
            anyhow::ensure!(
                fs::symlink_metadata(&target)?.is_dir(),
                "Invalid HF snapshot directory"
            );
        }
        target.push(relative.file_name().context("Missing HF merge filename")?);
        let new = journal.join(i.to_string());
        if *existed {
            let old = journal.join(format!("{i}.old"));
            if entry_exists(&old)? {
                fs::rename(old, &target)?;
            }
        } else if !entry_exists(&new)? {
            fs::rename(target, new)?;
        }
    }
    // The journal is replayable until it is retired in one rename.
    fs::rename(journal, retired)?;
    Ok(())
}

// Detect conflicts throughout an existing revision before replacing its first file.
// Entire new subdirectories can be installed with one rename.
fn plan_snapshot_merge(
    source: &Path,
    destination: &Path,
    may_replace: &impl Fn(&Path) -> bool,
    moves: &mut Vec<(PathBuf, PathBuf)>,
) -> anyhow::Result<()> {
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let target = destination.join(entry.file_name());
        let source_is_dir = entry.file_type()?.is_dir();
        match fs::symlink_metadata(&target) {
            Ok(existing) if source_is_dir && existing.is_dir() => {
                plan_snapshot_merge(&entry.path(), &target, may_replace, moves)?
            }
            Ok(existing) if source_is_dir || existing.is_dir() => {
                return Err(InvalidCacheDirectory(target).into());
            }
            Ok(existing) if !may_replace(&entry.path()) => {
                // A dangling pointer supplies no sibling and can be repaired.
                if existing.is_symlink() {
                    match fs::metadata(&target) {
                        Err(error) if error.kind() == ErrorKind::NotFound => {
                            moves.push((entry.path(), target))
                        }
                        Err(error) => return Err(error.into()),
                        Ok(_) => {}
                    }
                }
            }
            Err(error) if error.kind() != ErrorKind::NotFound => return Err(error.into()),
            _ => moves.push((entry.path(), target)),
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
    request: SnapshotRequest,
    streaming: bool,
    is_hf: bool,
) -> anyhow::Result<PathBuf> {
    let cache = get_model_express_cache_dir(provider);
    let ignore_weights = request.ignore_weights();
    // An online HF fallback writes refs/main itself. Isolate it until it can
    // publish under the same lock as a streamed download.
    if is_hf && !mx_envs::hf_offline() {
        let (staging, cache) = create_staging(&cache).await?;
        let staged_root = staging.path().to_owned();
        let repo = hf_repo_cache_dir(model_name);
        let (staging, revision) = if repo.join("snapshots").is_dir() {
            // Pin the remote commit before consulting any partial checkout.
            // Do not seed unrelated files or let a moving main select new bytes.
            let info = hf_hub::api::tokio::ApiBuilder::from_env()
                .with_token(mx_envs::hf_token())
                .with_cache_dir(staged_root.clone())
                .build()?
                .model(model_name.to_owned())
                .info()
                .await?;
            anyhow::ensure!(
                Path::new(&info.sha).file_name() == Some(OsStr::new(&info.sha)),
                "Invalid HF commit identifier"
            );
            let commit = info.sha.clone();
            let expected = request.clone();
            let stage_repo =
                staged_root.join(hf_hub::Repo::model(model_name.to_owned()).folder_name());
            let staging = tokio::task::spawn_blocking(move || {
                seed_hf_checkout(&repo, &stage_repo, &info, &expected)?;
                anyhow::Ok(staging)
            })
            .await??;
            (staging, Some(commit))
        } else {
            (staging, None)
        };
        let result = mx::download_model_revision(
            model_name,
            provider,
            Some(staged_root),
            ignore_weights,
            revision.as_deref(),
        )
        .await?;
        if let Some(commit) = &revision {
            anyhow::ensure!(
                result.resolved_revision.as_ref() == Some(commit),
                "HF returned a different commit than the requested revision"
            );
        }
        let blobs = staging
            .path()
            .join(hf_hub::Repo::model(model_name.to_owned()).folder_name())
            .join("blobs");
        return tokio::task::spawn_blocking(move || {
            materialize_hf_blob_links(&result.path, &blobs)?;
            publish_hf_snapshot(
                staging.path(),
                &result.path,
                &cache,
                result.resolved_revision.as_deref(),
                request,
                streaming,
            )
        })
        .await?;
    }

    // Offline mode only resolves a path in the real cache.
    let path = mx::download_model(model_name, provider, Some(cache), ignore_weights).await?;
    if is_hf {
        let check = (streaming && matches!(request, SnapshotRequest::Model { .. }))
            .then_some(ignore_weights);
        return local_hf_snapshot(model_name, path, check).await;
    }
    Ok(path)
}

// The provider's pinned hf-hub lookup reads refs/<commit> before resolving
// snapshots/<commit>/<file>. Seed only manifest entries it would request.
fn seed_hf_checkout(
    repo: &Path,
    stage_repo: &Path,
    info: &hf_hub::api::RepoInfo,
    request: &SnapshotRequest,
) -> anyhow::Result<()> {
    use modelexpress_common::providers::{
        ModelProviderTrait, huggingface::HuggingFaceProvider as Hf,
    };
    let cached = repo.join("snapshots").join(&info.sha);
    if ![repo, &repo.join("snapshots"), &cached]
        .iter()
        .all(|path| fs::symlink_metadata(path).is_ok_and(|m| m.is_dir()))
    {
        return Ok(());
    }
    let destination = stage_repo.join("snapshots").join(&info.sha);
    fs::create_dir_all(
        destination
            .parent()
            .context("Missing staged snapshots directory")?,
    )?;
    let seeded = read_hf_checkout(repo, || {
        let seed = tempfile::tempdir_in(stage_repo)?;
        let blob_dir = repo.join("blobs");
        let blobs = if fs::symlink_metadata(&blob_dir).is_ok_and(|m| m.is_dir()) {
            blob_dir.canonicalize().ok()
        } else {
            None
        };
        let refs = repo.join("refs");
        let published = [&info.sha[..], "main"]
            .iter()
            .any(|name| fs::read(refs.join(name)).is_ok_and(|value| value == info.sha.as_bytes()));
        // Every successful link pins immutable bytes; never copy large cached
        // files just to seed. A missing/unusable entry is downloaded normally.
        for f in &info.siblings {
            let name = f.rfilename.as_str();
            let rel = Path::new(name);
            if rel.file_name() != Some(OsStr::new(name))
                || Hf::is_ignored(name)
                || Hf::is_image(rel)
                || (request.ignore_weights() && Hf::is_weight_file(name))
            {
                continue;
            }
            let source = cached.join(rel);
            let Ok(meta) = fs::symlink_metadata(&source) else {
                continue;
            };
            let checksum = match request {
                SnapshotRequest::CheckedMetadata(files) => {
                    files.iter().find(|(n, _)| n == name).map(|(_, cf)| cf)
                }
                _ => None,
            };
            let file = if meta.is_symlink() {
                let Ok(target) = source.canonicalize() else {
                    continue;
                };
                if blobs.is_none() || target.parent() != blobs.as_deref() {
                    continue;
                }
                target
            } else if published || checksum.is_some() {
                source
            } else {
                // A loose, unverified file may belong to an interrupted transfer.
                continue;
            };
            if !fs::metadata(&file).is_ok_and(|m| m.is_file() && m.len() > 0) {
                continue;
            }
            let pinned = seed.path().join(rel);
            if fs::hard_link(&file, &pinned).is_err() {
                continue;
            }
            if let Some(expected) = checksum
                && crate::model_card::verify_metadata_file(&pinned, expected).is_err()
            {
                fs::remove_file(pinned)?;
            }
        }
        Ok(seed)
    })?;
    fs::rename(seeded.path(), destination)?;
    let refs = stage_repo.join("refs");
    fs::create_dir_all(&refs)?;
    fs::write(refs.join(&info.sha), &info.sha)?;
    Ok(())
}

// hf-hub uses relative links into the private repository's blobs directory.
// Keep their bytes when the snapshot is moved and the staging area is removed.
fn materialize_hf_blob_links(dir: &Path, blobs: &Path) -> anyhow::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let kind = entry.file_type()?;
        if kind.is_dir() {
            materialize_hf_blob_links(&path, blobs)?;
        } else if kind.is_symlink() {
            let source = path.canonicalize()?;
            anyhow::ensure!(
                source.parent() == Some(blobs.canonicalize()?.as_path())
                    && fs::metadata(&source)?.is_file(),
                "Invalid staged Hugging Face blob link: {}",
                path.display()
            );
            fs::remove_file(&path)?;
            if fs::hard_link(&source, &path).is_err() {
                fs::copy(&source, &path)
                    .with_context(|| format!("Failed to materialize {}", path.display()))?;
            }
        }
    }
    Ok(())
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

    const FULL_MODEL: SnapshotRequest = SnapshotRequest::Model {
        ignore_weights: false,
    };

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
            FULL_MODEL,
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
    fn checked_metadata_rejected_in_staging_preserves_existing_revision() {
        let cache = TempDir::new().unwrap();
        let staging = TempDir::new().unwrap();
        let model = "org/model";
        let existing = build_hf_cache(
            cache.path(),
            model,
            &["config.json", "tokenizer.json", "model.safetensors"],
        );
        let snapshot = build_hf_cache(staging.path(), model, &["config.json", "tokenizer.json"]);
        let names = ["config.json", "tokenizer.json"];
        let files = names.map(|name| {
            (
                name.to_owned(),
                CheckedFile::from_disk(existing.join(name)).unwrap(),
            )
        });
        fs::write(snapshot.join("tokenizer.json"), b"XX").unwrap();
        fs::write(snapshot.join("extra.json"), b"unpublished").unwrap();

        let err = publish_streamed_hf_snapshot(
            staging.path(),
            &snapshot,
            cache.path(),
            None,
            SnapshotRequest::CheckedMetadata(files.into()),
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("checksum mismatch"));
        assert_eq!(fs::read(existing.join("tokenizer.json")).unwrap(), b"{}");
        assert!(!existing.join("extra.json").exists());

        fs::write(snapshot.join("tokenizer.json"), b"{}").unwrap();
        fs::write(snapshot.join("config.json"), b"unchecked").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink("evicted-blob", existing.join("extra.json")).unwrap();
        let tokenizer = CheckedFile::from_disk(existing.join("tokenizer.json")).unwrap();
        let request =
            SnapshotRequest::CheckedMetadata([("tokenizer.json".to_owned(), tokenizer)].into());
        publish_streamed_hf_snapshot(staging.path(), &snapshot, cache.path(), None, request)
            .unwrap();
        assert_eq!(fs::read(existing.join("config.json")).unwrap(), b"{}");
        assert!(existing.join("extra.json").is_file());
        assert_eq!(
            get_cached_model_path_in(model, false, cache.path().into()).unwrap(),
            Some(existing)
        );
    }

    #[test]
    fn conflicting_cache_entries_are_rejected_before_publication() {
        let cache = TempDir::new().unwrap();
        let staging = TempDir::new().unwrap();
        let model = "org/model";
        let files = ["config.json", "tokenizer.json", "model.safetensors"];
        let existing = build_hf_cache(cache.path(), model, &files);
        let snapshot = build_hf_cache(staging.path(), model, &files);
        for name in files {
            fs::write(snapshot.join(name), b"replacement").unwrap();
        }
        fs::write(existing.join("nested"), b"keep").unwrap();
        fs::create_dir(snapshot.join("nested")).unwrap();
        let publish = |dir: &Path| {
            publish_streamed_hf_snapshot(staging.path(), dir, cache.path(), None, FULL_MODEL)
        };
        assert!(
            publish(&snapshot)
                .unwrap_err()
                .is::<InvalidCacheDirectory>()
        );
        for name in files {
            assert_eq!(fs::read(existing.join(name)).unwrap(), b"{}");
        }
        assert_eq!(
            get_cached_model_path_in(model, false, cache.path().into()).unwrap(),
            Some(existing.clone())
        );

        let main = existing
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("refs/main");
        fs::remove_file(&main).unwrap();
        fs::create_dir(&main).unwrap();
        let other = snapshot.with_file_name("other");
        fs::rename(snapshot, &other).unwrap();
        assert!(publish(&other).unwrap_err().is::<InvalidCacheDirectory>());
        assert!(!existing.with_file_name("other").exists());
        assert!(main.is_dir());
    }

    #[test]
    fn incomplete_stream_leaves_the_existing_snapshot_and_ref_untouched() {
        let cache = TempDir::new().unwrap();
        let staging = TempDir::new().unwrap();
        let model = "org/model";
        let existing = build_hf_cache(
            cache.path(),
            model,
            &["config.json", "tokenizer.json", "model.safetensors"],
        );
        let source = build_hf_cache(staging.path(), model, &["config.json", "tokenizer.json"]);
        fs::write(source.join("config.json"), b"replacement").unwrap();

        let publish = |snapshot: &Path| {
            publish_streamed_hf_snapshot(staging.path(), snapshot, cache.path(), None, FULL_MODEL)
        };
        assert!(
            publish(&source)
                .unwrap_err()
                .to_string()
                .contains("incomplete")
        );
        assert_eq!(fs::read(existing.join("config.json")).unwrap(), b"{}");
        let other = source.with_file_name("other");
        fs::rename(source, &other).unwrap();
        assert!(
            publish(&other)
                .unwrap_err()
                .to_string()
                .contains("incomplete")
        );
        assert!(!existing.with_file_name("other").exists());
        assert_eq!(
            get_cached_model_path_in(model, false, cache.path().into()).unwrap(),
            Some(existing)
        );
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
                SnapshotRequest::Model { ignore_weights },
            )
            .unwrap()
        };
        let full = publish("full-rev", false);
        let metadata = publish("metadata-rev", true);
        assert!(metadata.join("config.json").is_file());
        assert_eq!(
            get_cached_model_path_in(model, false, cache.path().into()).unwrap(),
            Some(full.clone())
        );

        fs::remove_file(full.join("model.safetensors")).unwrap();
        publish("metadata-rev", true);
        assert_eq!(
            get_cached_model_path_in(model, true, cache.path().into()).unwrap(),
            Some(metadata)
        );
    }

    #[test]
    fn streamed_snapshot_publish_failure_leaves_previous_ref() {
        let cache = TempDir::new().unwrap();
        let staging = TempDir::new_in(cache.path()).unwrap();
        let model = "org/model";
        let files = ["config.json", "tokenizer.json", "model.safetensors"];
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
            publish_streamed_hf_snapshot(staging.path(), &snapshot, cache.path(), None, FULL_MODEL)
                .is_err()
        );
        assert_eq!(
            get_cached_model_path_in(model, true, cache.path().to_path_buf()).unwrap(),
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
        let snapshot = build_hf_cache(
            staging.path(),
            model,
            &["config.json", "tokenizer.json", "model.safetensors"],
        );
        fs::write(snapshot.join("config.json"), b"new config").unwrap();
        fs::write(snapshot.join("model.safetensors"), b"complete").unwrap();

        assert_eq!(
            publish_streamed_hf_snapshot(staging.path(), &snapshot, cache.path(), None, FULL_MODEL)
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
            get_cached_model_path_in(model, false, cache.path().to_path_buf()).unwrap(),
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

        let snapshot = build_hf_cache(
            staging.path(),
            model,
            &["config.json", "tokenizer.json", "model.safetensors"],
        );
        fs::create_dir(snapshot.join("nested")).unwrap();
        fs::write(snapshot.join("nested/config.json"), b"streamed").unwrap();

        assert!(
            publish_streamed_hf_snapshot(staging.path(), &snapshot, cache.path(), None, FULL_MODEL)
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

        let with_weights =
            get_cached_model_path_in(model, false, temp.path().to_path_buf()).unwrap();
        let no_weights = get_cached_model_path_in(model, true, temp.path().to_path_buf()).unwrap();

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

    #[cfg(unix)]
    #[test]
    fn cached_model_tries_other_tokenizers_and_weights_after_io_errors() {
        let dir = TempDir::new().unwrap();
        let snapshot = build_hf_cache(
            dir.path(),
            "org/model",
            &["config.json", "pytorch_model.bin"],
        );
        for name in ["tokenizer.json", "model.safetensors", "broken.tiktoken"] {
            std::os::unix::fs::symlink(name, snapshot.join(name)).unwrap();
        }
        assert!(has_required_model_files(&snapshot, true).is_err());
        for alt in ["tiktoken.model", "valid.tiktoken"] {
            fs::write(snapshot.join(alt), b"tokenizer").unwrap();
            assert!(has_required_model_files(&snapshot, false).unwrap());
            fs::remove_file(snapshot.join(alt)).unwrap();
        }
    }

    #[test]
    fn native_metadata_cache_hit_still_requires_weights_for_full_fetch() {
        let cache = TempDir::new().unwrap();
        let model = "org/native";
        let snapshot = build_hf_cache(cache.path(), model, &["params.json", "tekken.json"]);
        assert_eq!(
            get_cached_model_path_in(model, true, cache.path().into()).unwrap(),
            Some(snapshot.clone())
        );
        assert!(
            get_cached_model_path_in(model, false, cache.path().into())
                .unwrap()
                .is_none()
        );
        fs::write(snapshot.join("consolidated.safetensors"), b"native weights").unwrap();
        assert_eq!(
            get_cached_model_path_in(model, false, cache.path().into()).unwrap(),
            Some(snapshot.clone())
        );
        // Some repositories ship HF configuration alongside their native artifacts.
        fs::write(snapshot.join("config.json"), "{}").unwrap();
        assert_eq!(
            get_cached_model_path_in(model, false, cache.path().into()).unwrap(),
            Some(snapshot.clone())
        );
        fs::remove_file(snapshot.join("params.json")).unwrap();
        fs::write(snapshot.join("tokenizer.json"), b"{}").unwrap();
        assert!(
            get_cached_model_path_in(model, false, cache.path().into())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn cache_lookup_rejects_directories_named_like_model_files() {
        let cache = TempDir::new().unwrap();
        let model = "org/model";
        let snapshot = build_hf_cache(cache.path(), model, &["config.json"]);
        fs::create_dir(snapshot.join("tokenizer.json")).unwrap();
        fs::create_dir(snapshot.join("tokenizer.tiktoken")).unwrap();
        assert!(
            get_cached_model_path_in(model, true, cache.path().to_path_buf())
                .unwrap()
                .is_none()
        );

        fs::remove_dir(snapshot.join("tokenizer.json")).unwrap();
        fs::write(snapshot.join("tokenizer.json"), b"{}").unwrap();
        fs::create_dir(snapshot.join("model.safetensors")).unwrap();
        assert!(
            get_cached_model_path_in(model, false, cache.path().to_path_buf())
                .unwrap()
                .is_none()
        );
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

        let with_weights =
            get_cached_model_path_in(model, false, temp.path().to_path_buf()).unwrap();
        let no_weights = get_cached_model_path_in(model, true, temp.path().to_path_buf()).unwrap();

        assert_eq!(with_weights.as_deref(), Some(snapshot.as_path()));
        assert_eq!(no_weights.as_deref(), Some(snapshot.as_path()));
    }

    #[test]
    fn weight_index_cannot_satisfy_its_own_shard() -> anyhow::Result<()> {
        let root = TempDir::new()?;
        let snap = build_hf_cache(
            root.path(),
            "org/sharded",
            &["config.json", "tokenizer.json"],
        );
        let index = snap.join("model.safetensors.index.json");
        fs::create_dir(snap.join("nested"))?;
        fs::write(&index, b"{}")?;
        fs::hard_link(&index, snap.join("nested/alias.safetensors"))?;
        for (name, expected) in [
            ("./model.safetensors.index.json", false),
            ("nested/alias.safetensors", false),
            ("nested/real.safetensors", true),
        ] {
            fs::write(
                &index,
                serde_json::to_vec(&serde_json::json!({"weight_map":{"a":name}}))?,
            )?;
            fs::write(snap.join("nested/real.safetensors"), b"weights")?;
            assert_eq!(has_required_model_files(&snap, false)?, expected, "{name}");
        }
        Ok(())
    }

    #[test]
    fn failed_late_merge_preserves_the_previous_checkout() -> anyhow::Result<()> {
        let root = TempDir::new()?;
        let cache = build_hf_cache(
            root.path(),
            "org/model",
            &["config.json", "tokenizer.json", "model.safetensors"],
        );
        let repo = root.path().join("models--org--model");
        let stage = TempDir::new_in(root.path())?;
        let source = stage.path().join("new-config");
        fs::write(&source, b"new")?;
        let extra = stage.path().join("added");
        fs::write(&extra, b"new sibling")?;
        // Simulate a type change after planning: a directory cannot replace a file.
        let invalid = stage.path().join("invalid");
        fs::create_dir(&invalid)?;
        let moves = vec![
            (source, cache.join("config.json")),
            (extra, cache.join("added")),
            (invalid, cache.join("tokenizer.json")),
        ];
        let journal = stage.path().join("live");
        assert!(install_snapshot_merge(stage.path(), &repo, &journal, &moves).is_err());
        assert!(!journal.exists());
        assert_eq!(fs::read(cache.join("config.json"))?, b"{}");
        assert!(!cache.join("added").exists());
        assert_eq!(
            get_cached_model_path_in("org/model", false, root.path().into()).unwrap(),
            Some(cache)
        );
        Ok(())
    }

    #[test]
    fn cache_reader_recovers_an_interrupted_same_revision_merge() -> anyhow::Result<()> {
        let root = TempDir::new()?;
        let model = "org/model";
        let snapshot = build_hf_cache(
            root.path(),
            model,
            &["config.json", "tokenizer.json", "model.safetensors"],
        );
        let repo = root.path().join("models--org--model");
        let name = repo.file_name().unwrap();
        let writer = lock_hf_publication(root.path(), name)?;
        let journal =
            hf_publication_lock_path(&root.path().join(".locks"), name).with_extension("merge");
        fs::create_dir(&journal)?;
        let config = snapshot.join("config.json");
        let extra = snapshot.join("added");
        fs::write(
            journal.join("manifest"),
            serde_json::to_vec(&[
                (config.strip_prefix(&repo)?, true),
                (extra.strip_prefix(&repo)?, false),
            ])?,
        )?;
        fs::rename(&config, journal.join("0.old"))?;
        fs::write(&config, "new config")?;
        fs::write(&extra, "new sibling")?;
        drop(writer); // OS releases this lock when a publishing process exits.
        assert_eq!(
            get_cached_model_path_in(model, false, root.path().to_owned()).unwrap(),
            Some(snapshot)
        );
        assert_eq!(fs::read(config)?, b"{}");
        assert!(!extra.exists());
        assert!(!journal.exists());
        fs::create_dir(&journal)?;
        fs::write(
            journal.join("manifest"),
            b"interrupted or unreadable journal",
        )?;
        assert!(get_cached_model_path_in(model, false, root.path().to_owned()).is_err());
        Ok(())
    }

    #[test]
    fn checked_reader_retries_when_first_publisher_creates_its_lock() -> anyhow::Result<()> {
        let root = TempDir::new()?;
        let repo = root.path().join("models--org--model");
        let snapshot = repo.join("snapshots/r");
        fs::create_dir_all(&snapshot)?;
        let config = snapshot.join("config.json");
        let tokenizer = snapshot.join("tokenizer.json");
        fs::write(&config, "old")?;
        fs::write(&tokenizer, "old")?;
        let n = std::cell::Cell::new(0);
        let read = read_hf_checkout(&repo, || {
            let first = fs::read(&config)?;
            n.set(n.get() + 1);
            if n.get() == 1 {
                let _writer = lock_hf_publication(root.path(), repo.file_name().unwrap())?;
                fs::write(&config, "new")?;
                fs::write(&tokenizer, "new")?;
            } else {
                let lock_path = hf_publication_lock_path(
                    &root.path().join(".locks"),
                    repo.file_name().unwrap(),
                );
                assert!(matches!(
                    open_cache_lock(&lock_path, false)?.try_lock(),
                    Err(fs::TryLockError::WouldBlock)
                ));
            }
            Ok((first, fs::read(&tokenizer)?))
        })?;
        assert_eq!(n.get(), 2);
        assert_eq!(read, (b"new".to_vec(), b"new".to_vec()));
        Ok(())
    }

    #[tokio::test]
    #[serial_test::serial]
    #[cfg(unix)]
    async fn direct_fallback_skips_weights_cached_for_the_resolved_commit() -> anyhow::Result<()> {
        let root = TempDir::new()?;
        let cache = root.path().join("hub");
        let mut hub = mockito::Server::new_async().await;
        let endpoint = hub.url();
        let model = "org/partial";
        let current = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let old = build_hf_cache(&cache, model, &["config.json"]);
        let partial = old.with_file_name(current);
        fs::create_dir_all(&partial)?;
        let blobs = cache.join("models--org--partial/blobs");
        fs::create_dir_all(&blobs)?;
        fs::write(blobs.join("weight-blob"), b"cached weights")?;
        std::os::unix::fs::symlink("../../blobs/weight-blob", partial.join("model.safetensors"))?;
        fs::write(partial.join("tokenizer.json"), b"truncated")?;
        let siblings = ["config.json", "tokenizer.json", "model.safetensors"]
            .map(|name| serde_json::json!({"rfilename":name}));
        let info = serde_json::json!({"id":model, "sha":current, "siblings":siblings});
        let branch_lookup = hub
            .mock("GET", "/api/models/org/partial/revision/main")
            .with_body(info.to_string())
            .create_async()
            .await;
        let commit_lookup = hub
            .mock(
                "GET",
                format!("/api/models/{model}/revision/{current}").as_str(),
            )
            .with_body(info.to_string())
            .create_async()
            .await;
        let weights = hub
            .mock(
                "GET",
                format!("/{model}/resolve/{current}/model.safetensors").as_str(),
            )
            .expect(0)
            .create_async()
            .await;
        let mut missing = Vec::new();
        for name in ["config.json", "tokenizer.json"] {
            missing.push(
                hub.mock("GET", format!("/{model}/resolve/{current}/{name}").as_str())
                    .with_header("x-repo-commit", current)
                    .with_header("etag", name)
                    .with_body("{}")
                    .expect_at_least(1)
                    .create_async()
                    .await,
            );
        }
        temp_env::async_with_vars(
            [
                ("HOME", Some(root.path().as_os_str())),
                ("HF_HUB_CACHE", Some(cache.as_os_str())),
                ("HF_HUB_OFFLINE", None),
                ("HF_ENDPOINT", Some(endpoint.as_ref())),
                ("MODEL_EXPRESS_NO_SHARED_STORAGE", Some("1".as_ref())),
                ("MODEL_EXPRESS_URL", Some("http://[".as_ref())),
            ],
            async {
                assert_eq!(from_hf(model, false).await?, partial);
                assert_eq!(
                    fs::read(partial.join("model.safetensors"))?,
                    b"cached weights"
                );
                assert_eq!(fs::read(partial.join("tokenizer.json"))?, b"{}");
                branch_lookup.assert_async().await;
                commit_lookup.assert_async().await;
                weights.assert_async().await;
                for file in missing {
                    file.assert_async().await;
                }
                anyhow::Ok(())
            },
        )
        .await
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

        let incomplete = get_cached_model_path_in(model, false, temp.path().to_path_buf()).unwrap();
        assert!(
            incomplete.is_none(),
            "sharded cache without shard files must NOT satisfy ignore_weights=false"
        );

        fs::write(snapshot.join("model-00001-of-00002.safetensors"), "").unwrap();
        fs::write(snapshot.join("model-00002-of-00002.safetensors"), "").unwrap();
        let complete = get_cached_model_path_in(model, false, temp.path().to_path_buf()).unwrap();
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
            get_cached_model_path_in(model, true, temp.path().to_path_buf())
                .unwrap()
                .is_none(),
            "tokenizer_config.json alone must NOT satisfy ignore_weights=true",
        );
        assert!(
            get_cached_model_path_in(model, false, temp.path().to_path_buf())
                .unwrap()
                .is_none(),
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
                            .unwrap()
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
