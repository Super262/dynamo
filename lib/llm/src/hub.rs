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
const HF_WEIGHT_INDEXES: [&str; 2] = [
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
];

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
            let is_hf = MxModelProvider::resolve_provider_for_model_name(model_name, provider)
                == MxModelProvider::HuggingFace;
            if !is_hf {
                return get_cached_model_path_in(model_name, ignore_weights, cache_dir);
            }
            let offline = is_offline_mode();
            first_cache_hit(hf_cache_roots().map(|cache| {
                // A shared ModelExpress checkout may have no HF main ref.
                if offline || cache != cache_dir {
                    let repo = cache.join(hf_hub::Repo::model(model_name.to_owned()).folder_name());
                    read_hf_checkout(&repo, || {
                        complete_hf_checkout(&repo, model_name, ignore_weights)
                    })
                } else {
                    get_cached_model_path_in(model_name, ignore_weights, cache)
                }
            }))
        }
        _ => Ok(get_cached_provider_model_path_in(
            model_name,
            provider,
            ignore_weights,
            &cache_dir,
        )),
    }
}

// A hit in another candidate can satisfy the request despite an earlier I/O error.
pub(super) fn first_cache_hit<T>(
    candidates: impl IntoIterator<Item = anyhow::Result<Option<T>>>,
) -> anyhow::Result<Option<T>> {
    let mut first_error = None;
    for candidate in candidates {
        match candidate {
            Ok(Some(hit)) => return Ok(Some(hit)),
            Ok(None) => {}
            Err(error) => {
                first_error.get_or_insert(error);
            }
        }
    }
    first_error.map_or(Ok(None), Err)
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
        match snapshot.and_then(|path| path.parent().map(Path::to_owned)) {
            Some(path) => cached_hf_model_dir(&repo_dir, model_name, path, ignore_weights),
            None => Ok(None),
        }
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

fn cached_file_type(path: &Path) -> io::Result<Option<fs::FileType>> {
    match fs::metadata(path) {
        Ok(metadata) => Ok(Some(metadata.file_type())),
        Err(error) if matches!(error.kind(), ErrorKind::NotFound | ErrorKind::NotADirectory) => {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

fn cached_file(path: &Path) -> io::Result<bool> {
    Ok(cached_file_type(path)?.is_some_and(|kind| kind.is_file()))
}

/// Require a configuration and tokenizer in the same model format, plus weights
/// unless explicitly skipped. Preserve I/O errors for callers updating refs.
fn has_required_model_files(dir: &Path, ignore_weights: bool) -> io::Result<bool> {
    let has = |filename: &str| cached_file(&dir.join(filename));
    // The loader opens an existing config.json even when native metadata is available.
    let has_config = match cached_file_type(&dir.join("config.json"))? {
        Some(kind) if kind.is_file() => true,
        Some(_) => return Ok(false),
        None => false,
    };
    let native_metadata = || Ok(has("params.json")? && has("tekken.json")?);
    let hf_metadata = || {
        let tokenizers = ["tokenizer.json", "tiktoken.model"].into_iter().map(has);
        Ok(has_config
            && any_available(tokenizers.chain(std::iter::once_with(|| has_tiktoken_file(dir))))?)
    };
    if !any_available(
        std::iter::once_with(hf_metadata).chain(std::iter::once_with(&native_metadata)),
    )? {
        return Ok(false);
    }
    if ignore_weights {
        return Ok(true);
    }

    has_model_weights(dir, native_metadata)
}

fn has_model_weights(
    dir: &Path,
    native_layout: impl FnOnce() -> io::Result<bool>,
) -> io::Result<bool> {
    let has = |filename: &str| cached_file(&dir.join(filename));
    let weights = ["model.safetensors", "pytorch_model.bin"]
        .into_iter()
        .map(has)
        .chain(std::iter::once_with(|| {
            Ok(has("consolidated.safetensors")? && native_layout()?)
        }));
    let indexes = HF_WEIGHT_INDEXES
        .into_iter()
        .map(|name| shard_files_present(&dir.join(name)));
    any_available(weights.chain(indexes))
}

// Direct downloads resolve and fetch the repository's advertised artifacts.
// The backend may use tokenizer files that Dynamo itself does not recognize.
fn has_direct_model_files(snapshot: &Path, files: &[String]) -> io::Result<bool> {
    match cached_file_type(&snapshot.join("config.json"))? {
        Some(kind) if kind.is_file() => {}
        Some(_) => return Ok(false),
        None if cached_file(&snapshot.join("params.json"))? => {}
        None => return Ok(false),
    }
    if !has_model_weights(snapshot, || cached_file(&snapshot.join("params.json")))? {
        return Ok(false);
    }
    for name in files {
        if Path::new(name).file_name() != Some(OsStr::new(name))
            || !cached_file(&snapshot.join(name))?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn direct_model_artifacts(snapshot: &Path) -> io::Result<Vec<String>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(snapshot)? {
        let entry = entry?;
        if let Ok(name) = entry.file_name().into_string()
            && cached_file(&entry.path())?
        {
            files.push(name);
        }
    }
    files.sort_unstable();
    Ok(files)
}

fn cached_hf_model_files(
    repo: &Path,
    snapshot: &Path,
    ignore_weights: bool,
) -> anyhow::Result<bool> {
    let standard = has_required_model_files(snapshot, ignore_weights);
    if matches!(&standard, Ok(true)) {
        return Ok(true);
    }
    let delegated = (|| {
        let cache = repo.parent().context("HF repository has no cache root")?;
        let name = repo.file_name().context("HF repository has no name")?;
        let state = hf_publication_lock_path(&cache.join(".locks"), name).with_extension("order");
        let order = HfRequestOrder::read(&state)?;
        let Some(sidecars) = snapshot
            .file_name()
            .and_then(OsStr::to_str)
            .and_then(|revision| order.direct_models.get(revision))
        else {
            return Ok(false);
        };
        Ok(has_direct_model_files(snapshot, sidecars)?)
    })();
    if matches!(&delegated, Ok(true)) {
        return Ok(true);
    }
    standard?;
    delegated
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
    let Some(shards) = index_shards(&fs::read(index_path)?) else {
        return Ok(false);
    };
    for shard in shards {
        let path = snapshot_dir.join(shard);
        if !cached_file(&path)? || same_file::is_same_file(index_path, &path)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn index_shards(contents: &[u8]) -> Option<std::collections::BTreeSet<PathBuf>> {
    let value: serde_json::Value = serde_json::from_slice(contents).ok()?;
    let weight_map = value.get("weight_map")?.as_object()?;
    if weight_map.is_empty() {
        return None;
    }
    weight_map
        .values()
        .map(|shard| normalized_hf_path(shard.as_str()?))
        .collect()
}

fn normalized_hf_path(name: &str) -> Option<PathBuf> {
    if name.contains('\\') || name.ends_with('/') || name.ends_with("/.") {
        return None;
    }
    let mut path = PathBuf::new();
    for component in Path::new(name).components() {
        match component {
            Component::Normal(name) => path.push(name),
            Component::CurDir => {}
            _ => return None,
        }
    }
    (!path.as_os_str().is_empty()).then_some(path)
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
    Model {
        ignore_weights: bool,
    },
    CheckedMetadata {
        files: Arc<[(String, CheckedFile)]>,
        blobs: PathBuf,
    },
}

impl SnapshotRequest {
    fn ignore_weights(&self) -> bool {
        match self {
            Self::Model { ignore_weights } => *ignore_weights,
            Self::CheckedMetadata { .. } => true,
        }
    }
}

pub(super) fn hf_cache_roots() -> impl Iterator<Item = PathBuf> {
    let preferred = get_model_express_cache_dir(MxModelProvider::HuggingFace);
    let shared = (!is_no_shared_storage())
        .then(|| MxClientConfig::default().cache.local_path)
        .filter(|root| *root != preferred);
    std::iter::once(preferred).chain(shared)
}

fn hf_repo_cache_dir(repo: &str) -> PathBuf {
    get_model_express_cache_dir(MxModelProvider::HuggingFace)
        .join(hf_hub::Repo::model(repo.to_owned()).folder_name())
}

pub(super) async fn snapshot_for_checked_metadata(
    repo: &str,
    files: Arc<[(String, CheckedFile)]>,
    blobs: PathBuf,
) -> anyhow::Result<PathBuf> {
    fetch_snapshot(
        Path::new(repo),
        SnapshotRequest::CheckedMetadata { files, blobs },
    )
    .await
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
    // Both transfer attempts publish on behalf of this fetch. Shared-storage
    // hits and offline reads need no ticket until they actually stage a write.
    let mut ticket = None;
    let result = match mx_download_from_server(
        &model_name,
        provider,
        request.clone(),
        config,
        hf_stream_cache.as_deref(),
        &mut ticket,
    )
    .await
    {
        Ok(path) => Ok(path),
        // Both downloaders publish into the same HF cache.
        Err(error)
            if error.is::<InvalidCacheDirectory>() || error.is::<CacheInspectionFailed>() =>
        {
            Err(error)
        }
        Err(server_error) => {
            tracing::warn!(%model_name, error = %server_error, "ModelExpress fetch failed, falling back to direct download");
            mx_download_direct(&model_name, provider, request, hf_stream_cache.is_some(), is_hf, &mut ticket).await
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
    request: SnapshotRequest,
    mut config: MxClientConfig,
    hf_stream_cache: Option<&Path>,
    ticket: &mut Option<u64>,
) -> anyhow::Result<PathBuf> {
    let staging = if let Some(cache) = hf_stream_cache {
        let (staging, cache, order) = create_staging(cache, model_name, ticket).await?;
        config.cache.local_path = staging.path().to_path_buf();
        Some((staging, cache, order))
    } else {
        None
    };

    let mut client = MxClient::new(config).await?;
    tracing::info!("Successfully connected to ModelExpress server");
    let download = client
        .request_model_revision(model_name, provider, request.ignore_weights(), None)
        .await?;
    tracing::info!("Server download succeeded for model: {model_name}");

    if let Some((staging, cache, order)) = staging {
        let snapshot = download
            .path
            .context("ModelExpress returned no streamed snapshot")?;
        return tokio::task::spawn_blocking(move || {
            publish_hf_snapshot(
                staging.path(),
                &snapshot,
                &cache,
                download.resolved_revision.as_deref(),
                request,
                HfModelValidation::Stream,
                order,
            )
        })
        .await?;
    }

    // In shared storage the server already resolved the immutable revision.
    // Use that checkout even if this client cannot contact the upstream Hub.
    let is_hf = MxModelProvider::resolve_provider_for_model_name(model_name, provider)
        == MxModelProvider::HuggingFace;
    if !is_hf {
        return client.get_model_path(model_name, provider).await;
    }
    if let Some(rev) = download.resolved_revision {
        let folder = hf_hub::Repo::model(model_name.to_owned()).folder_name();
        return tokio::task::spawn_blocking(move || {
            shared_hf_revision(&folder, &rev, &request, download.path.as_deref())
        })
        .await?;
    }

    // Older servers may omit the revision; let the provider select the checkout.
    let hf = mx::get_provider(MxModelProvider::HuggingFace);
    let mut first_error = None;
    for root in hf_cache_roots() {
        let result = match hf.get_model_path(model_name, root).await {
            Ok(path) => local_hf_snapshot(path).await,
            Err(error) => Err(error),
        };
        match result {
            Ok(path) => return Ok(path),
            Err(error) => {
                first_error.get_or_insert(error);
            }
        }
    }
    Err(first_error.unwrap_or_else(|| anyhow::anyhow!("No shared Hugging Face cache root")))
}

// Try a complete checkout in both roots before accepting a provider-defined
// layout. A checked card can also supply missing slots from verified MDC blobs.
fn shared_hf_revision(
    folder: &str,
    rev: &str,
    request: &SnapshotRequest,
    reported: Option<&Path>,
) -> anyhow::Result<PathBuf> {
    anyhow::ensure!(
        Path::new(rev).file_name() == Some(OsStr::new(rev)),
        "Invalid Hugging Face revision returned by ModelExpress"
    );
    let mut provider_path = None;
    let complete = first_cache_hit(hf_cache_roots().map(|root| {
        let repo = root.join(folder);
        let path = repo.join("snapshots").join(rev);
        let candidate = read_hf_checkout(&repo, || {
            if !cached_file_type(&path)?.is_some_and(|kind| kind.is_dir()) {
                return Ok(None);
            }
            match request {
                SnapshotRequest::Model { ignore_weights } => {
                    if has_required_model_files(&path, *ignore_weights)? {
                        return Ok(Some(true));
                    }
                    let eligible =
                        *ignore_weights || !metadata_only_revisions(&root, &repo)?.contains(rev);
                    Ok(eligible.then_some(false))
                }
                SnapshotRequest::CheckedMetadata { files, blobs } => Ok(Some(
                    crate::model_card::matching_metadata_files(&path, files, Some(blobs)).is_ok(),
                )),
            }
        })?;
        match candidate {
            Some(true) => return Ok(Some(path)),
            Some(false) if provider_path.is_none() || reported == Some(path.as_path()) => {
                provider_path = Some(path);
            }
            _ => {}
        }
        Ok(None)
    }))
    .context(CacheInspectionFailed)?;
    complete
        .or(provider_path)
        .with_context(|| format!("ModelExpress revision {rev} is unavailable in the shared cache"))
}

#[derive(Debug, thiserror::Error)]
#[error("Failed to inspect shared Hugging Face cache")]
struct CacheInspectionFailed;

pub(super) fn hf_snapshot_repo(path: &Path) -> anyhow::Result<&Path> {
    path.parent()
        .filter(|dir| dir.file_name() == Some(OsStr::new("snapshots")))
        .and_then(Path::parent)
        .with_context(|| format!("Invalid HF snapshot directory: {}", path.display()))
}

async fn local_hf_snapshot(path: PathBuf) -> anyhow::Result<PathBuf> {
    tokio::task::spawn_blocking(move || {
        let exists = read_hf_checkout(hf_snapshot_repo(&path)?, || {
            Ok(cached_file_type(&path)?.is_some_and(|kind| kind.is_dir()))
        })
        .context(CacheInspectionFailed)?;
        anyhow::ensure!(
            exists,
            "Missing Hugging Face snapshot at {}",
            path.display()
        );
        Ok(path)
    })
    .await?
}

async fn create_staging(
    cache: &Path,
    model: &str,
    ticket: &mut Option<u64>,
) -> anyhow::Result<(StreamStaging, PathBuf, u64)> {
    let repo = hf_hub::Repo::model(model.to_owned()).folder_name();
    let cache = cache.to_path_buf();
    let previous = *ticket;
    let result = tokio::task::spawn_blocking(move || {
        let staging = StreamStaging::new(&cache)
            .with_context(|| format!("Failed to stage a download in {}", cache.display()))?;
        let order = match previous {
            Some(order) => order,
            None => reserve_hf_request(&cache, OsStr::new(&repo), staging.path())?,
        };
        anyhow::Ok((staging, cache, order))
    })
    .await??;
    *ticket = Some(result.2);
    Ok(result)
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

// Logical manifest entry for the per-repo order file beside the journal in .locks.
const HF_ORDER_FILE: &str = ".dynamo-main-order";

#[derive(Default, serde::Deserialize, serde::Serialize)]
struct HfRequestOrder {
    issued: u64,
    main: Option<PublishedMain>,
    #[serde(default)]
    metadata_only: Option<std::collections::BTreeSet<String>>,
    #[serde(default)]
    direct_models: std::collections::BTreeMap<String, Vec<String>>,
}

#[derive(serde::Deserialize, serde::Serialize)]
struct PublishedMain {
    ticket: u64,
    revision: String,
    kind: PublishedSnapshotKind,
}

#[derive(serde::Deserialize, serde::Serialize)]
enum PublishedSnapshotKind {
    Metadata,
    Full,
    DirectFull,
    ProviderDefinedFull,
}

#[derive(Clone, Copy)]
enum HfModelValidation {
    Stream,
    Direct,
    SharedProvider,
}

impl PublishedMain {
    fn should_keep(
        &self,
        ticket: u64,
        ignore_weights: bool,
        main_ref: &Path,
        snapshots: &Path,
        direct_sidecars: Option<&[String]>,
    ) -> io::Result<bool> {
        let full = !matches!(self.kind, PublishedSnapshotKind::Metadata);
        // Prefer a complete checkout, then the later request within that class.
        if (full, self.ticket) <= (!ignore_weights, ticket) {
            return Ok(false);
        }
        match fs::read(main_ref) {
            Ok(main) if main == self.revision.as_bytes() => {}
            Ok(_) => return Ok(false),
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(e),
        }
        let path = snapshots.join(&self.revision);
        match self.kind {
            PublishedSnapshotKind::Full => has_required_model_files(&path, false),
            PublishedSnapshotKind::DirectFull => any_available(
                std::iter::once_with(|| has_required_model_files(&path, false)).chain(
                    direct_sidecars
                        .into_iter()
                        .map(|sidecars| has_direct_model_files(&path, sidecars)),
                ),
            ),
            _ => match fs::metadata(&path) {
                Ok(meta) => Ok(meta.is_dir()),
                Err(e) if e.kind() == ErrorKind::NotFound => Ok(false),
                Err(e) => Err(e),
            },
        }
    }
}

impl HfRequestOrder {
    fn read(path: &Path) -> anyhow::Result<Self> {
        match fs::read(path) {
            Ok(bytes) => {
                let mut order: Self = serde_json::from_slice(&bytes)?;
                if order.metadata_only.is_none() {
                    let legacy = order
                        .main
                        .iter()
                        .filter(|main| matches!(main.kind, PublishedSnapshotKind::Metadata))
                        .map(|main| main.revision.clone());
                    order.metadata_only = Some(legacy.collect());
                }
                Ok(order)
            }
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e.into()),
        }
    }
}

// MX returns the revision after the transfer. Order overlapping resolutions by
// when they start, using the shared cache lock rather than clocks on different hosts.
fn reserve_hf_request(cache: &Path, repo: &OsStr, staging: &Path) -> anyhow::Result<u64> {
    let _lock = lock_hf_publication(cache, repo)?;
    let state = hf_publication_lock_path(&cache.join(".locks"), repo).with_extension("order");
    let mut order = HfRequestOrder::read(&state)?;
    order.issued = order
        .issued
        .checked_add(1)
        .context("HF request counter exhausted")?;
    let next = staging.join("next-order");
    fs::write(&next, serde_json::to_vec(&order)?)?;
    fs::rename(next, state)?;
    Ok(order.issued)
}

fn publish_hf_snapshot(
    staging: &Path,
    snapshot: &Path,
    cache: &Path,
    resolved_revision: Option<&str>,
    request: SnapshotRequest,
    validation: HfModelValidation,
    ticket: u64,
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

    let mut direct_sidecars = None;
    match &request {
        SnapshotRequest::Model { .. }
            if matches!(validation, HfModelValidation::SharedProvider) => {}
        SnapshotRequest::Model { ignore_weights } => {
            let complete = has_required_model_files(snapshot, *ignore_weights);
            let delegated = !ignore_weights && matches!(validation, HfModelValidation::Direct);
            let complete = if delegated && !matches!(&complete, Ok(true)) {
                let sidecars = direct_model_artifacts(snapshot)?;
                let matches = has_direct_model_files(snapshot, &sidecars)?;
                if matches {
                    direct_sidecars = Some(sidecars);
                }
                matches
            } else {
                complete.with_context(|| {
                    format!("Failed to inspect staged snapshot {}", snapshot.display())
                })?
            };
            anyhow::ensure!(
                complete,
                "Staged Hugging Face snapshot at {} is incomplete",
                snapshot.display()
            );
        }
        SnapshotRequest::CheckedMetadata { files, blobs } => {
            crate::model_card::matching_metadata_files(snapshot, files, Some(blobs))
                .context("Downloaded metadata does not match the model card")?;
        }
    }

    let cache_root = cache.canonicalize()?;
    // Recover a journal left by eviction before recreating the repository.
    let _lock = lock_hf_publication(&cache_root, repo)?;
    let repo_dir = cache_child_dir(&cache_root, repo)?;
    let refs = cache_child_dir(&repo_dir, OsStr::new("refs"))?;
    let snapshots = cache_child_dir(&repo_dir, OsStr::new("snapshots"))?;
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
    let mut moves = Vec::new();
    let new_revision = match fs::symlink_metadata(&destination) {
        Ok(metadata) if metadata.is_dir() => {
            // A native-only update leaves an existing config.json in place.
            if !matches!(validation, HfModelValidation::SharedProvider)
                && matches!(request, SnapshotRequest::Model { .. })
                && !cached_file(&snapshot.join("config.json"))?
                && cached_file_type(&destination.join("config.json"))?
                    .is_some_and(|kind| !kind.is_file())
            {
                return Err(InvalidCacheDirectory(destination.join("config.json")).into());
            }
            // A staged omission is allowed when the MDC has a verified blob.
            // An existing, conflicting leaf would still be visible after merging.
            if let SnapshotRequest::CheckedMetadata { files, .. } = &request {
                for (name, expected) in files.iter() {
                    if cached_file(&snapshot.join(name))? {
                        continue;
                    }
                    if cached_file_type(&destination.join(name))?.is_some() {
                        crate::model_card::verify_metadata_file(&destination.join(name), expected)
                            .context("Existing snapshot conflicts with sparse checked metadata")?;
                    }
                }
            }
            // A checked card vouches only for its advertised files.
            let may_replace = |path: &Path| match &request {
                SnapshotRequest::CheckedMetadata { files, .. } => {
                    files.iter().any(|(name, _)| path == snapshot.join(name))
                }
                SnapshotRequest::Model { .. } => true,
            };
            plan_snapshot_merge(snapshot, &destination, &may_replace, &mut moves)?;
            false
        }
        Ok(_) => return Err(InvalidCacheDirectory(destination).into()),
        Err(e) if e.kind() == ErrorKind::NotFound => {
            // A cold revision still becomes visible with one directory rename.
            moves.push((snapshot.to_owned(), destination));
            true
        }
        Err(e) => return Err(e.into()),
    };
    if publish_pinned {
        moves.push((pinned_ref, pinned_target));
    }

    let journal =
        hf_publication_lock_path(&cache_root.join(".locks"), repo).with_extension("merge");
    let state = journal.with_extension("order");
    let mut order = HfRequestOrder::read(&state)?;
    let newer_main = match &order.main {
        Some(main) => main.should_keep(
            ticket,
            request.ignore_weights(),
            &main_ref,
            &snapshots,
            order.direct_models.get(&main.revision).map(Vec::as_slice),
        )?,
        None => false,
    };
    let preserve_full = request.ignore_weights() && main_has_complete_snapshot(&main_ref, &snapshots)
        .unwrap_or_else(|error| {
            tracing::warn!(path = %main_ref.display(), %error, "Preserving the current ref because snapshot validation failed");
            true
        });
    let revision = revision.to_str().context("Invalid HF revision encoding")?;
    let metadata_only = order.metadata_only.get_or_insert_default();
    // A metadata merge into an existing provider checkout does not establish
    // that its preexisting contents are metadata-only.
    let provenance_changed = if request.ignore_weights() {
        new_revision && metadata_only.insert(revision.to_owned())
    } else {
        metadata_only.remove(revision)
    };
    let mut direct_changed = false;
    let kind = if request.ignore_weights() {
        PublishedSnapshotKind::Metadata
    } else {
        let before = order.direct_models.len();
        order.direct_models.retain(|revision, _| {
            !fs::symlink_metadata(snapshots.join(revision))
                .is_err_and(|error| error.kind() == ErrorKind::NotFound)
        });
        direct_changed = order.direct_models.len() != before;
        match (validation, direct_sidecars) {
            (HfModelValidation::Direct, Some(sidecars)) => {
                order.direct_models.insert(revision.to_owned(), sidecars);
                direct_changed = true;
                PublishedSnapshotKind::DirectFull
            }
            (HfModelValidation::SharedProvider, _) => PublishedSnapshotKind::ProviderDefinedFull,
            _ => {
                direct_changed |= order.direct_models.remove(revision).is_some();
                PublishedSnapshotKind::Full
            }
        }
    };
    let update_main = !newer_main && !preserve_full;
    if update_main {
        let staged_ref = staging.join("main-ref");
        fs::write(&staged_ref, revision.as_bytes())?;
        moves.push((staged_ref, main_ref));
        order.main = Some(PublishedMain {
            ticket,
            revision: revision.to_owned(),
            kind,
        });
    }
    if update_main || provenance_changed || direct_changed {
        let staged_order = staging.join("main-order");
        fs::write(&staged_order, serde_json::to_vec(&order)?)?;
        moves.push((staged_order, state));
    }
    install_snapshot_merge(staging, &repo_dir, &journal, &moves)?;
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
        drop(captured);
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
        let relative = if *target == journal.with_extension("order") {
            Path::new(HF_ORDER_FILE)
        } else {
            target.strip_prefix(repo)?
        };
        entries.push((relative.to_owned(), existed));
        fs::rename(source, prepared.join(i.to_string()))?;
    }
    fs::write(prepared.join("manifest"), serde_json::to_vec(&entries)?)?;
    fs::rename(&prepared, journal)?;
    let install = (|| -> io::Result<()> {
        for (i, ((_, target), (_, existed))) in moves.iter().zip(&entries).enumerate() {
            if *existed {
                let old = journal.join(format!("{i}.old"));
                // Keep the name readable across replacement when hard links are
                // supported. Renaming also works on protected multi-user caches.
                if fs::hard_link(target, &old).is_err() {
                    fs::rename(target, old)?;
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

fn cache_directory_exists(path: &Path) -> anyhow::Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() => Ok(true),
        Ok(_) => Err(InvalidCacheDirectory(path.to_owned()).into()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

// Ref contents become path components during recovery, including backups from an
// older native cache. A missing ref is distinct from a ref that cannot be read.
fn merge_ref_revision(path: &Path) -> anyhow::Result<Option<String>> {
    if !entry_exists(path)? {
        return Ok(None);
    }
    anyhow::ensure!(
        fs::metadata(path)?.is_file(),
        "Invalid HF ref at {}",
        path.display()
    );
    let value = fs::read_to_string(path)?;
    let revision = value.trim();
    anyhow::ensure!(
        Path::new(revision).file_name() == Some(OsStr::new(revision)),
        "Invalid HF revision in {}",
        path.display(),
    );
    Ok(Some(revision.to_owned()))
}

fn rollback_snapshot_merge(repo: &Path, journal: &Path, retired: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        fs::symlink_metadata(journal)?.is_dir(),
        "Invalid HF merge journal"
    );
    let entries: Vec<(PathBuf, bool)> =
        serde_json::from_slice(&fs::read(journal.join("manifest"))?)?;
    let evicted = !cache_directory_exists(repo)?;
    let mut revisions = std::collections::BTreeSet::new();
    let mut new_snapshots = std::collections::BTreeSet::new();
    let mut backup_refs = std::collections::BTreeMap::new();
    let mut replay = Vec::with_capacity(entries.len());
    for (i, (relative, existed)) in entries.iter().enumerate() {
        let components: Vec<_> = relative.components().collect();
        anyhow::ensure!(
            (matches!(components.as_slice(), [Component::Normal(prefix), Component::Normal(_), ..] if *prefix == "snapshots")
                || matches!(components.as_slice(), [Component::Normal(prefix), Component::Normal(_)] if *prefix == "refs")
                || relative == Path::new(HF_ORDER_FILE))
                && components.iter().all(|c| matches!(c, Component::Normal(_))),
            "Invalid path in HF merge journal"
        );
        if !evicted
            && let [Component::Normal(prefix), Component::Normal(rev), tail @ ..] =
                components.as_slice()
            && *prefix == "snapshots"
            && (!tail.is_empty() || !entry_exists(&journal.join(i.to_string()))?)
        {
            // A missing cold snapshot that never left staging was not evicted.
            revisions.insert(rev.to_os_string());
            if tail.is_empty() && !existed {
                new_snapshots.insert(rev.to_os_string());
            }
        }
        if !evicted && relative.starts_with("refs") {
            // If the replacement never left staging and the old name is still
            // in place, this transaction did not change the ref (even a stale one).
            if entry_exists(&journal.join(i.to_string()))?
                && (!*existed || entry_exists(&repo.join(relative))?)
            {
                continue;
            }
            if *existed && let Some(rev) = merge_ref_revision(&journal.join(format!("{i}.old")))? {
                revisions.insert(std::ffi::OsString::from(&rev));
                backup_refs.insert(i, rev);
            }
        }
        replay.push((i, relative, existed));
    }
    let snapshots = repo.join("snapshots");
    let mut missing = std::collections::BTreeSet::new();
    if !revisions.is_empty() {
        anyhow::ensure!(
            cache_directory_exists(&snapshots)?,
            "Missing HF snapshots directory"
        );
    }
    for revision in revisions {
        if !cache_directory_exists(&snapshots.join(&revision))? {
            missing.insert(revision);
        }
    }
    let ref_unavailable = |rev: &str| {
        let rev = OsStr::new(rev);
        missing.contains(rev) || new_snapshots.contains(rev)
    };
    let repair_order = !missing.is_empty() || backup_refs.values().any(|rev| ref_unavailable(rev));
    let mut keep_main = None;
    let mut staged_order = None;
    for (i, relative, existed) in replay.into_iter().rev() {
        let components: Vec<_> = relative.components().collect();
        if (evicted && relative != Path::new(HF_ORDER_FILE))
            || matches!(components.as_slice(), [Component::Normal(prefix), Component::Normal(rev), ..]
                if *prefix == "snapshots" && missing.contains(*rev))
        {
            continue;
        }
        let mut target = repo.to_owned();
        for component in &components[..components.len() - 1] {
            target.push(component);
            anyhow::ensure!(
                fs::symlink_metadata(&target)?.is_dir(),
                "Invalid HF snapshot directory"
            );
        }
        if relative == Path::new(HF_ORDER_FILE) {
            target = journal.with_extension("order");
        } else {
            target.push(relative.file_name().context("Missing HF merge filename")?);
        }
        let new = journal.join(i.to_string());
        let old = journal.join(format!("{i}.old"));
        if repair_order && relative == Path::new(HF_ORDER_FILE) {
            // Preserve the attempted order through a second interruption. It
            // may own the only surviving main after the old target was evicted.
            if *existed && entry_exists(&old)? && !entry_exists(&new)? && entry_exists(&target)? {
                fs::rename(&target, &new)?;
            }
            staged_order = Some(new.clone());
        }
        if repair_order
            && relative.starts_with("refs")
            && backup_refs.get(&i).is_none_or(|rev| ref_unavailable(rev))
        {
            if let Some(current) = merge_ref_revision(&target)? {
                if !ref_unavailable(&current) && cache_directory_exists(&snapshots.join(&current))?
                {
                    if relative == Path::new("refs/main")
                        && backup_refs.get(&i).is_some_and(|rev| ref_unavailable(rev))
                    {
                        keep_main = Some(current);
                    }
                } else {
                    fs::remove_file(&target)?;
                }
            }
            continue;
        }
        if *existed {
            // A failed install can leave the unchanged target and its hard-link
            // backup in place, including when the target directory is read-only.
            if entry_exists(&old)? && (!entry_exists(&new)? || !entry_exists(&target)?) {
                fs::rename(old, &target)?;
            }
        } else if !entry_exists(&new)? {
            fs::rename(target, new)?;
        }
    }
    if evicted || repair_order {
        let state = journal.with_extension("order");
        let mut order = HfRequestOrder::read(&state)?;
        // Preserve issued tickets and provenance for surviving revisions.
        if evicted
            || order
                .main
                .as_ref()
                .is_some_and(|main| ref_unavailable(&main.revision))
        {
            order.main = None;
        }
        if let Some(metadata) = &mut order.metadata_only {
            metadata.retain(|revision| !evicted && !ref_unavailable(revision));
        }
        order
            .direct_models
            .retain(|revision, _| !evicted && !ref_unavailable(revision));
        if let (Some(current), Some(staged)) = (keep_main, staged_order)
            && let Some(published) = HfRequestOrder::read(&staged)?.main
            && published.revision == current
        {
            order.main = Some(published);
        }
        let tmp = retired.with_file_name("evicted-order");
        fs::write(&tmp, serde_json::to_vec(&order)?)?;
        fs::rename(tmp, state)?;
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
    ticket: &mut Option<u64>,
) -> anyhow::Result<PathBuf> {
    let cache = get_model_express_cache_dir(provider);
    let ignore_weights = request.ignore_weights();
    // An online HF fallback writes refs/main itself. Isolate it until it can
    // publish under the same lock as a streamed download.
    if is_hf && !mx_envs::hf_offline() {
        let (staging, cache, order) = create_staging(&cache, model_name, ticket).await?;
        let staged_root = staging.path().to_owned();
        let repo = hf_repo_cache_dir(model_name);
        let manifest = if repo.join("snapshots").is_dir() {
            // Resolve main and request LFS checksums in the same Hub query.
            let result = async {
                let info: HfSeedManifest = hf_hub::api::tokio::ApiBuilder::from_env()
                    .with_token(mx_envs::hf_token())
                    .with_cache_dir(staged_root.clone())
                    .build()?
                    .model(model_name.to_owned())
                    .info_request()
                    .query(&[("blobs", "true")])
                    .send()
                    .await?
                    .error_for_status()?
                    .json()
                    .await?;
                anyhow::Ok(info)
            }
            .await;
            match result {
                Ok(info) => Some(info),
                Err(error) => {
                    tracing::warn!(%model_name, %error, "Hugging Face file manifest unavailable, downloading without seeding");
                    None
                }
            }
        } else {
            None
        };
        let stage_repo = staged_root.join(hf_hub::Repo::model(model_name.to_owned()).folder_name());
        let (staging, manifest) = if let Some(info) = manifest {
            anyhow::ensure!(
                Path::new(&info.sha).file_name() == Some(OsStr::new(&info.sha)),
                "Invalid HF commit identifier"
            );
            let expected = request.clone();
            let source = repo.clone();
            let destination = stage_repo.clone();
            tokio::task::spawn_blocking(move || {
                seed_hf_checkout(&source, &destination, &info, &expected)?;
                anyhow::Ok((staging, Some(info)))
            })
            .await??
        } else {
            (staging, None)
        };
        let revision = manifest.as_ref().map(|info| info.sha.as_str());
        let result = mx::download_model_revision(
            model_name,
            provider,
            Some(staged_root),
            ignore_weights,
            revision,
        )
        .await?;
        if let Some(commit) = revision {
            anyhow::ensure!(
                result.resolved_revision.as_deref() == Some(commit)
                    && result.path == stage_repo.join("snapshots").join(commit),
                "HF returned a different checkout than the requested revision"
            );
        }
        return tokio::task::spawn_blocking(move || {
            materialize_hf_blob_links(&result.path, &stage_repo.join("blobs"))?;
            if !ignore_weights && let Some(info) = &manifest {
                recover_indexed_hf_shards(&repo, &result.path, info)?;
            }
            publish_hf_snapshot(
                staging.path(),
                &result.path,
                &cache,
                result.resolved_revision.as_deref(),
                request,
                if streaming {
                    HfModelValidation::Direct
                } else {
                    HfModelValidation::SharedProvider
                },
                order,
            )
        })
        .await?;
    }

    // Offline mode only resolves a path in the real cache.
    if is_hf && matches!(request, SnapshotRequest::Model { .. }) {
        let model = model_name.to_owned();
        return tokio::task::spawn_blocking(move || {
            let folder = hf_hub::Repo::model(model.clone()).folder_name();
            first_cache_hit(hf_cache_roots().map(|cache| {
                let repo = cache.join(&folder);
                read_hf_checkout(&repo, || {
                    offline_cached_model(&cache, &repo, &model, ignore_weights, !streaming)
                })
            }))?
            .with_context(|| format!("Cached Hugging Face snapshots for {model} are incomplete"))
        })
        .await?;
    }
    let path = mx::download_model(model_name, provider, Some(cache), ignore_weights).await?;
    if is_hf {
        return local_hf_snapshot(path).await;
    }
    Ok(path)
}

fn metadata_only_revisions(
    cache: &Path,
    repo: &Path,
) -> anyhow::Result<std::collections::BTreeSet<String>> {
    let name = repo.file_name().context("HF repository has no name")?;
    let state = hf_publication_lock_path(&cache.join(".locks"), name).with_extension("order");
    Ok(HfRequestOrder::read(&state)?
        .metadata_only
        .unwrap_or_default())
}

fn offline_cached_model(
    cache: &Path,
    repo: &Path,
    model_name: &str,
    ignore_weights: bool,
    allow_provider_layout: bool,
) -> anyhow::Result<Option<PathBuf>> {
    if let Some(path) = complete_hf_checkout(repo, model_name, ignore_weights)? {
        return Ok(Some(path));
    }

    if allow_provider_layout {
        let metadata_only = metadata_only_revisions(cache, repo)?;
        for entry in recent_hf_snapshots(&repo.join("snapshots"))? {
            let entry = entry?;
            if ignore_weights
                || !entry
                    .file_name()
                    .to_str()
                    .is_some_and(|rev| metadata_only.contains(rev))
            {
                return Ok(Some(entry.path()));
            }
        }
    }
    Ok(None)
}

fn cached_hf_model_dir(
    repo: &Path,
    model_name: &str,
    path: PathBuf,
    ignore_weights: bool,
) -> anyhow::Result<Option<PathBuf>> {
    if cached_hf_model_files(repo, &path, ignore_weights)? {
        tracing::info!("Found cached model '{model_name}' at {path:?}, skipping download");
        Ok(Some(path))
    } else {
        Ok(None)
    }
}

fn complete_hf_checkout(
    repo: &Path,
    model_name: &str,
    ignore_weights: bool,
) -> anyhow::Result<Option<PathBuf>> {
    let snapshots = repo.join("snapshots");
    let main = fs::read_to_string(repo.join("refs/main"))
        .ok()
        .filter(|revision| Path::new(revision).file_name() == Some(OsStr::new(revision)))
        .map(|revision| snapshots.join(revision));
    let check = |path| cached_hf_model_dir(repo, model_name, path, ignore_weights);
    let preferred = main.as_ref().map(|path| match check(path.clone()) {
        // A metadata request may publish another revision alongside an unreadable
        // current snapshot. The publisher preserves main when it cannot inspect it.
        Err(error)
            if ignore_weights
                && error
                    .downcast_ref::<io::Error>()
                    .is_some_and(|e| e.kind() == ErrorKind::PermissionDenied)
                && fs::read_dir(path).is_err_and(|e| e.kind() == ErrorKind::PermissionDenied) =>
        {
            Ok(None)
        }
        other => other,
    });
    first_cache_hit(preferred.into_iter().chain(std::iter::once_with(|| {
        let entries = recent_hf_snapshots(&snapshots)
            .with_context(|| format!("Reading Hugging Face snapshots for {model_name}"))?;
        first_cache_hit(entries.map(|entry| {
            let path = entry?.path();
            if Some(&path) == main.as_ref() {
                Ok(None)
            } else {
                check(path)
            }
        }))
    })))
}

// Preserve ModelExpress's preference for newer cached revisions. Break
// timestamp ties by revision name to get the same order on repeated reads.
pub(super) fn recent_hf_snapshots(
    snapshots: &Path,
) -> io::Result<impl Iterator<Item = io::Result<fs::DirEntry>>> {
    let entries = match fs::read_dir(snapshots) {
        Ok(entries) => Some(entries),
        Err(error) if error.kind() == ErrorKind::NotFound => None,
        Err(error) => return Err(error),
    };
    Ok(order_hf_snapshots(entries.into_iter().flatten()))
}

fn order_hf_snapshots(
    entries: impl Iterator<Item = io::Result<fs::DirEntry>>,
) -> impl Iterator<Item = io::Result<fs::DirEntry>> {
    let mut candidates = Vec::new();
    let mut first_error = None;
    for entry in entries {
        match entry.and_then(|entry| Ok(entry.file_type()?.is_dir().then_some(entry))) {
            Ok(Some(dir)) => candidates.push(dir),
            Ok(None) => {}
            Err(error) => {
                first_error.get_or_insert(error);
            }
        }
    }
    candidates.sort_by_cached_key(|entry| {
        let time = entry
            .metadata()
            .and_then(|meta| meta.created().or_else(|_| meta.modified()))
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        (std::cmp::Reverse(time), entry.file_name())
    });
    candidates.into_iter().map(Ok).chain(first_error.map(Err))
}

#[derive(serde::Deserialize)]
struct HfSeedManifest {
    sha: String,
    siblings: Vec<HfSeedFile>,
}

#[derive(serde::Deserialize)]
struct HfSeedFile {
    rfilename: String,
    #[serde(rename = "blobId")]
    blob_id: Option<String>,
    lfs: Option<HfLfsFile>,
}

impl HfSeedFile {
    fn matches(&self, path: &Path) -> io::Result<bool> {
        if let Some(lfs) = &self.lfs {
            return lfs.matches(path);
        }
        let Some(blob_id) = &self.blob_id else {
            return Ok(false);
        };
        if blob_id.len() != 40 || !blob_id.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Ok(false);
        }

        use sha1::{Digest, Sha1};
        let file = fs::File::open(path)?;
        let size = file.metadata()?.len();
        let mut hasher = Sha1::new();
        hasher.update(format!("blob {size}\0"));
        let bytes = io::copy(&mut io::BufReader::new(file), &mut hasher)?;
        Ok(bytes == size && format!("{:x}", hasher.finalize()).eq_ignore_ascii_case(blob_id))
    }
}

#[derive(serde::Deserialize)]
struct HfLfsFile {
    sha256: String,
    size: u64,
}

impl HfLfsFile {
    fn matches(&self, path: &Path) -> io::Result<bool> {
        use sha2::{Digest, Sha256};
        let file = fs::File::open(path)?;
        if file.metadata()?.len() != self.size {
            return Ok(false);
        }
        let mut file = io::BufReader::new(file);
        let mut hasher = Sha256::new();
        let size = io::copy(&mut file, &mut hasher)?;
        Ok(size == self.size && format!("{:x}", hasher.finalize()) == self.sha256)
    }
}

// The provider's pinned hf-hub lookup reads refs/<commit> before resolving
// snapshots/<commit>/<file>. ModelExpress only requests top-level files.
fn seed_hf_checkout(
    repo: &Path,
    stage_repo: &Path,
    info: &HfSeedManifest,
    request: &SnapshotRequest,
) -> anyhow::Result<()> {
    use modelexpress_common::providers::{
        ModelProviderTrait, huggingface::HuggingFaceProvider as Hf,
    };
    let Some(cached) = cached_hf_seed_snapshot(repo, &info.sha) else {
        return Ok(());
    };
    let destination = stage_repo.join("snapshots").join(&info.sha);
    fs::create_dir_all(
        destination
            .parent()
            .context("Missing staged snapshots directory")?,
    )?;
    let (seeded, verified_candidates) = read_hf_checkout(repo, || {
        let seed = tempfile::tempdir_in(stage_repo)?;
        let mut verified_candidates = Vec::new();
        let blobs = cached_hf_seed_blobs(repo);
        // Hard links keep the selected inode available across cache renames.
        for f in &info.siblings {
            let name = f.rfilename.as_str();
            let Some(rel) = normalized_hf_path(name) else {
                continue;
            };
            if rel.components().count() != 1
                || Hf::is_ignored(name)
                || Hf::is_image(&rel)
                || (request.ignore_weights() && Hf::is_weight_file(name))
            {
                continue;
            }
            let source = cached.join(&rel);
            let Some(file) = cached_seed_file(&source, blobs.as_deref()) else {
                continue;
            };
            let checksum = match request {
                SnapshotRequest::CheckedMetadata { files, .. } => {
                    files.iter().find(|(n, _)| n == name).map(|(_, cf)| cf)
                }
                _ => None,
            };
            if checksum.is_none() && f.lfs.is_none() && f.blob_id.is_none() {
                continue;
            }
            let Ok(metadata) = fs::metadata(&file) else {
                continue;
            };
            if !metadata.is_file()
                || metadata.len() == 0
                || f.lfs.as_ref().is_some_and(|lfs| lfs.size != metadata.len())
            {
                continue;
            }
            let pinned = seed.path().join(&rel);
            if fs::hard_link(&file, &pinned).is_err() {
                continue;
            }
            if let Some(expected) = checksum {
                if crate::model_card::verify_metadata_file(&pinned, expected).is_err() {
                    fs::remove_file(pinned)?;
                }
            } else {
                verified_candidates.push((pinned, f));
            }
        }
        Ok((seed, verified_candidates))
    })?;
    // Verify the Hub manifest identity after releasing the publication lock.
    for (path, file) in verified_candidates {
        if !file.matches(&path).unwrap_or(false) {
            fs::remove_file(path)?;
        }
    }
    fs::rename(seeded.path(), destination)?;
    let refs = stage_repo.join("refs");
    fs::create_dir_all(&refs)?;
    fs::write(refs.join(&info.sha), &info.sha)?;
    Ok(())
}

// ModelExpress skips nested Hub files. Recover only shards required by the
// index in the private checkout, which may have been absent from the live cache.
fn recover_indexed_hf_shards(
    repo: &Path,
    snapshot: &Path,
    info: &HfSeedManifest,
) -> anyhow::Result<()> {
    use modelexpress_common::providers::{
        ModelProviderTrait, huggingface::HuggingFaceProvider as Hf,
    };

    let mut needed = std::collections::BTreeSet::new();
    for name in HF_WEIGHT_INDEXES {
        let index = snapshot.join(name);
        if cached_file(&index)?
            && let Some(shards) = index_shards(&fs::read(index)?)
        {
            for shard in shards {
                if shard.components().count() > 1 && !entry_exists(&snapshot.join(&shard))? {
                    needed.insert(shard);
                }
            }
        }
    }
    if needed.is_empty() {
        return Ok(());
    }

    let (pins, linked) = read_hf_checkout(repo, || {
        let pins = tempfile::tempdir_in(
            snapshot
                .parent()
                .context("Missing staged snapshots directory")?,
        )?;
        let mut linked = Vec::new();
        if let Some(cached) = cached_hf_seed_snapshot(repo, &info.sha) {
            let blobs = cached_hf_seed_blobs(repo);
            for (i, file) in info.siblings.iter().enumerate() {
                let Some(lfs) = &file.lfs else {
                    continue;
                };
                let Some(path) = normalized_hf_path(&file.rfilename) else {
                    continue;
                };
                if lfs.size == 0
                    || !needed.contains(&path)
                    || Hf::is_ignored(&file.rfilename)
                    || Hf::is_image(&path)
                {
                    continue;
                }
                let parent = path.parent().context("Missing shard parent")?;
                if !parent
                    .ancestors()
                    .take_while(|p| !p.as_os_str().is_empty())
                    .all(|p| fs::symlink_metadata(cached.join(p)).is_ok_and(|meta| meta.is_dir()))
                {
                    continue;
                }
                let Some(source) = cached_seed_file(&cached.join(&path), blobs.as_deref()) else {
                    continue;
                };
                let pin = pins.path().join(i.to_string());
                if fs::metadata(&source).is_ok_and(|meta| meta.is_file() && meta.len() == lfs.size)
                    && fs::hard_link(&source, &pin).is_ok()
                {
                    linked.push((path, pin, lfs));
                }
            }
        }
        Ok((pins, linked))
    })?;
    // The pins keep the source inodes available while hashing outside the lock.
    for (path, pin, lfs) in linked {
        let destination = snapshot.join(path);
        if lfs.matches(&pin).unwrap_or(false) && !entry_exists(&destination)? {
            fs::create_dir_all(
                destination
                    .parent()
                    .context("Missing staged shard parent")?,
            )?;
            fs::rename(pin, destination)?;
        }
    }
    drop(pins);
    Ok(())
}

fn cached_hf_seed_snapshot(repo: &Path, revision: &str) -> Option<PathBuf> {
    let snapshots = repo.join("snapshots");
    let snapshot = snapshots.join(revision);
    [repo, &snapshots, &snapshot]
        .iter()
        .all(|path| fs::symlink_metadata(path).is_ok_and(|meta| meta.is_dir()))
        .then_some(snapshot)
}

fn cached_hf_seed_blobs(repo: &Path) -> Option<PathBuf> {
    let blobs = repo.join("blobs");
    if fs::symlink_metadata(&blobs).is_ok_and(|meta| meta.is_dir()) {
        blobs.canonicalize().ok()
    } else {
        None
    }
}

fn cached_seed_file(path: &Path, blobs: Option<&Path>) -> Option<PathBuf> {
    let meta = fs::symlink_metadata(path).ok()?;
    if meta.is_file() {
        return Some(path.to_owned());
    }
    if !meta.is_symlink() {
        return None;
    }
    let target = path.canonicalize().ok()?;
    if target.parent()? == blobs? && target.is_file() {
        Some(target)
    } else {
        None
    }
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

    fn publish_streamed_hf_snapshot(
        staging: &Path,
        snapshot: &Path,
        cache: &Path,
        resolved_revision: Option<&str>,
        request: SnapshotRequest,
    ) -> anyhow::Result<PathBuf> {
        publish_hf_snapshot(
            staging,
            snapshot,
            cache,
            resolved_revision,
            request,
            HfModelValidation::Stream,
            0,
        )
    }

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
        let blobs = cache.path().join("blobs");
        fs::create_dir(&blobs).unwrap();
        fs::copy(
            existing.join("tokenizer.json"),
            blobs.join(files[1].1.checksum().hash()),
        )
        .unwrap();

        let err = publish_streamed_hf_snapshot(
            staging.path(),
            &snapshot,
            cache.path(),
            None,
            SnapshotRequest::CheckedMetadata {
                files: files.into(),
                blobs,
            },
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
        let request = SnapshotRequest::CheckedMetadata {
            files: [("tokenizer.json".to_owned(), tokenizer)].into(),
            blobs: cache.path().join("blobs"),
        };
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
    fn sparse_checked_publication_requires_an_identifying_slot_and_verified_blobs()
    -> anyhow::Result<()> {
        let cache = TempDir::new()?;
        let staging = TempDir::new()?;
        let model = "org/model";
        let staged = build_hf_cache(staging.path(), model, &["extra.json"]);
        let blobs = cache.path().join("blobs");
        fs::create_dir(&blobs)?;
        let config = staging.path().join("config-source");
        let generation = staging.path().join("generation-source");
        fs::write(&config, b"config")?;
        fs::write(&generation, b"generation")?;
        let files = [
            ("config.json".to_owned(), CheckedFile::from_disk(&config)?),
            (
                "generation_config.json".to_owned(),
                CheckedFile::from_disk(&generation)?,
            ),
        ];
        fs::copy(config, blobs.join(files[0].1.checksum().hash()))?;
        let generation_blob = blobs.join(files[1].1.checksum().hash());
        fs::copy(&generation, &generation_blob)?;
        let request = SnapshotRequest::CheckedMetadata {
            files: files.into(),
            blobs,
        };
        let publish = |dir: &Path, source: &Path| {
            publish_streamed_hf_snapshot(dir, source, cache.path(), None, request.clone())
        };

        let error = publish(staging.path(), &staged).unwrap_err();
        assert!(format!("{error:#}").contains("No advertised metadata file identifies"));
        assert!(!cache.path().join("models--org--model").exists());
        fs::write(staged.join("config.json"), b"config")?;
        fs::write(&generation_blob, b"corrupt")?;
        assert!(publish(staging.path(), &staged).is_err());
        assert!(!cache.path().join("models--org--model").exists());
        fs::copy(generation, generation_blob)?;

        let checkout = publish(staging.path(), &staged)?;
        assert!(checkout.join("extra.json").is_file());
        assert!(!checkout.join("generation_config.json").exists());

        // The existing leaf is also part of the final merged snapshot. An
        // absent staged slot cannot disguise a conflicting destination file.
        fs::write(
            checkout.join("generation_config.json"),
            b"conflicting bytes",
        )?;
        let more = build_hf_cache(staging.path(), model, &["new-sibling.json"]);
        fs::write(more.join("config.json"), b"config")?;
        let error = publish(staging.path(), &more).unwrap_err();
        assert!(format!("{error:#}").contains("Existing snapshot conflicts"));
        assert!(!checkout.join("new-sibling.json").exists());
        assert_eq!(
            fs::read(checkout.join("generation_config.json"))?,
            b"conflicting bytes"
        );
        Ok(())
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
    fn latest_successful_full_request_owns_main_in_either_completion_order() -> anyhow::Result<()> {
        for older_finishes_first in [false, true] {
            let cache = TempDir::new()?;
            let name = OsStr::new("models--org--model");
            let start = |revision| -> anyhow::Result<_> {
                let stage = TempDir::new_in(cache.path())?;
                let order = reserve_hf_request(cache.path(), name, stage.path())?;
                let snapshot = build_hf_cache(
                    stage.path(),
                    "org/model",
                    &["config.json", "tokenizer.json", "model.safetensors"],
                );
                let dest = snapshot.with_file_name(revision);
                fs::rename(snapshot, &dest)?;
                Ok((stage, dest, order))
            };
            let old = start("earlier")?;
            let new = start("later")?;
            let _failed = start("never-published")?;
            let requests = if older_finishes_first {
                [old, new]
            } else {
                [new, old]
            };
            for (stage, snapshot, order) in requests {
                let revision = snapshot.file_name().unwrap();
                let actual = publish_hf_snapshot(
                    stage.path(),
                    &snapshot,
                    cache.path(),
                    None,
                    FULL_MODEL,
                    HfModelValidation::Stream,
                    order,
                )?;
                assert_eq!(actual.file_name(), Some(revision));
                assert!(actual.join("model.safetensors").is_file());
            }
            assert_eq!(
                fs::read(cache.path().join(name).join("refs/main"))?,
                b"later"
            );
        }
        Ok(())
    }

    #[test]
    fn existing_full_main_uses_its_own_layout_when_requests_overlap() -> anyhow::Result<()> {
        let root = TempDir::new()?;
        let main = root.path().join("main");
        fs::write(&main, "custom")?;
        fs::create_dir(root.path().join("custom"))?;
        let mut published = PublishedMain {
            revision: "custom".to_owned(),
            ticket: 2,
            kind: PublishedSnapshotKind::ProviderDefinedFull,
        };
        assert!(published.should_keep(1, false, &main, root.path(), None)?);
        assert!(published.should_keep(3, true, &main, root.path(), None)?);
        published.kind = PublishedSnapshotKind::Full;
        assert!(!published.should_keep(1, false, &main, root.path(), None)?);
        fs::remove_dir(root.path().join("custom"))?;
        published.kind = PublishedSnapshotKind::ProviderDefinedFull;
        assert!(!published.should_keep(1, false, &main, root.path(), None)?);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn rollback_leaves_an_uninstalled_ref_in_its_readonly_directory() -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let root = TempDir::new()?;
        let repo = root.path().join("models--org--model");
        let snapshot = repo.join("snapshots/B");
        let refs = repo.join("refs");
        fs::create_dir_all(&snapshot)?;
        fs::create_dir(&refs)?;
        let main = refs.join("main");
        fs::write(&main, "preexisting-missing-revision")?;
        fs::write(snapshot.join("config.json"), "replacement")?;
        fs::write(snapshot.join("model.safetensors"), "addition")?;

        let journal = root.path().join("transaction");
        fs::create_dir(&journal)?;
        fs::write(
            journal.join("manifest"),
            serde_json::to_vec(&[
                ("snapshots/B/config.json", true),
                ("snapshots/B/model.safetensors", false),
                ("refs/main", true),
            ])?,
        )?;
        fs::write(journal.join("0.old"), "original")?;
        fs::write(journal.join("2"), "B")?;
        fs::hard_link(&main, journal.join("2.old"))?;
        let saved_permissions = fs::metadata(&refs)?.permissions();
        fs::set_permissions(&refs, fs::Permissions::from_mode(0o555))?;
        let result = rollback_snapshot_merge(&repo, &journal, &root.path().join("retired"));
        fs::set_permissions(&refs, saved_permissions)?;
        result?;
        assert!(!journal.exists());
        assert_eq!(fs::read(main)?, b"preexisting-missing-revision");
        assert_eq!(fs::read(snapshot.join("config.json"))?, b"original");
        assert!(!snapshot.join("model.safetensors").exists());
        Ok(())
    }

    #[test]
    fn failed_main_ref_restores_pinned_ref_and_snapshot() -> anyhow::Result<()> {
        for existing in [false, true] {
            let cache = TempDir::new()?;
            let repo = cache.path().join("models--org--model");
            let snapshots = repo.join("snapshots");
            let refs = repo.join("refs");
            fs::create_dir_all(snapshots.join("old"))?;
            fs::create_dir(snapshots.join("previous pin"))?;
            fs::create_dir(&refs)?;
            fs::write(refs.join("main"), "old")?;
            fs::write(refs.join("next"), "previous pin")?;
            let stage = TempDir::new_in(cache.path())?;
            let dest = snapshots.join("next");
            let source = stage.path().join("source");
            let leaf = if existing {
                fs::create_dir(&dest)?;
                fs::write(dest.join("config.json"), "old config")?;
                fs::write(&source, "new config")?;
                dest.join("config.json")
            } else {
                fs::create_dir(&source)?;
                fs::write(source.join("config.json"), "new config")?;
                dest.clone()
            };
            let pin = stage.path().join("pin");
            fs::write(&pin, "next")?;
            let invalid = stage.path().join("invalid");
            fs::create_dir(&invalid)?; // Force the final rename to fail over a regular ref.
            let journal = cache.path().join("transaction");
            let moves = [
                (source, leaf),
                (pin, refs.join("next")),
                (invalid, refs.join("main")),
            ];
            assert!(install_snapshot_merge(stage.path(), &repo, &journal, &moves).is_err());
            assert!(!journal.exists());
            assert_eq!(fs::read(refs.join("main"))?, b"old");
            assert_eq!(fs::read(refs.join("next"))?, b"previous pin");
            if existing {
                assert_eq!(fs::read(dest.join("config.json"))?, b"old config");
            } else {
                assert!(!dest.exists());
            }
        }
        Ok(())
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
        fs::create_dir(snapshot.join("config.json")).unwrap();
        for ignore_weights in [false, true] {
            assert!(
                get_cached_model_path_in(model, ignore_weights, cache.path().into())
                    .unwrap()
                    .is_none()
            );
        }
        let staging = TempDir::new().unwrap();
        let update = build_hf_cache(
            staging.path(),
            model,
            &["params.json", "tekken.json", "consolidated.safetensors"],
        );
        assert!(
            publish_streamed_hf_snapshot(staging.path(), &update, cache.path(), None, FULL_MODEL)
                .is_err()
        );
        assert_eq!(
            fs::read(snapshot.join("consolidated.safetensors")).unwrap(),
            b"native weights"
        );
        fs::remove_dir(snapshot.join("config.json")).unwrap();
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

    #[tokio::test]
    #[serial_test::serial]
    async fn shared_hf_cache_prefers_hf_root_and_reuses_modelexpress_root() -> anyhow::Result<()> {
        let root = TempDir::new()?;
        let home = root.path().join("home");
        let hf = root.path().join("hf");
        let model = "org/shared";
        let files = ["config.json", "tokenizer.json", "model.safetensors"];
        temp_env::async_with_vars(
            [
                ("HOME", Some(home.as_os_str())),
                ("HF_HUB_CACHE", Some(hf.as_os_str())),
                ("HF_HUB_OFFLINE", Some("1".as_ref())),
                ("MODEL_EXPRESS_NO_SHARED_STORAGE", None),
                ("MODEL_EXPRESS_URL", Some("http://[".as_ref())),
            ],
            async {
                let mx = MxClientConfig::default().cache.local_path;
                assert!(mx.starts_with(&home));
                let shared = build_hf_cache(&mx, model, &files);
                assert_eq!(from_hf(model, false).await?, shared);
                fs::rename(
                    shared.join("model.safetensors"),
                    shared.join("custom.weights"),
                )?;
                assert_eq!(from_hf(model, false).await?, shared);
                fs::rename(
                    shared.join("custom.weights"),
                    shared.join("model.safetensors"),
                )?;
                build_hf_cache(&hf, model, &["config.json"]);
                let folder = hf_hub::Repo::model(model.to_owned()).folder_name();
                let rev = shared.file_name().unwrap().to_str().unwrap();
                assert_eq!(
                    shared_hf_revision(&folder, rev, &FULL_MODEL, Some(&shared))?,
                    shared
                );
                let preferred = build_hf_cache(&hf, model, &files);
                assert_eq!(from_hf(model, false).await?, preferred);

                let lock = hf_publication_lock_path(&hf.join(".locks"), OsStr::new(&folder));
                fs::create_dir_all(&lock)?;
                assert_eq!(from_hf(model, false).await?, shared);
                fs::remove_file(shared.join("model.safetensors"))?;
                let error = from_hf(model, false).await.unwrap_err();
                assert!(
                    format!("{error:#}").contains("Invalid HF publication lock"),
                    "{error:#}"
                );
                assert!(lock.is_dir());
                anyhow::Ok(())
            },
        )
        .await
    }

    #[test]
    fn alternate_directory_scan_retains_an_entry_error_until_a_hit() -> anyhow::Result<()> {
        let root = TempDir::new()?;
        let usable = root.path().join("revision");
        fs::create_dir(&usable)?;
        let scan = || {
            order_hf_snapshots(
                std::iter::once(Err(io::Error::from(ErrorKind::PermissionDenied)))
                    .chain(fs::read_dir(root.path()).unwrap()),
            )
        };
        let lookup = |wanted: &Path| {
            first_cache_hit(scan().map(|entry| {
                let path = entry?.path();
                Ok((path == wanted).then_some(path))
            }))
        };
        assert_eq!(lookup(&usable)?, Some(usable));
        assert_eq!(
            lookup(&root.path().join("missing"))
                .unwrap_err()
                .downcast_ref::<io::Error>()
                .unwrap()
                .kind(),
            ErrorKind::PermissionDenied
        );
        Ok(())
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
            ("nested/real.safetensors/", false),
            ("nested/real.safetensors/.", false),
            ("./nested/real.safetensors", true),
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
        let cold = repo.join("snapshots/new-rev");
        let pin = repo.join("refs/new-rev");
        let main = repo.join("refs/main");
        let order = journal.with_extension("order");
        fs::write(&order, b"original order")?;
        fs::write(
            journal.join("manifest"),
            serde_json::to_vec(&[
                (config.strip_prefix(&repo)?, true),
                (extra.strip_prefix(&repo)?, false),
                (cold.strip_prefix(&repo)?, false),
                (pin.strip_prefix(&repo)?, false),
                (main.strip_prefix(&repo)?, true),
                (Path::new(HF_ORDER_FILE), true),
            ])?,
        )?;
        fs::rename(&config, journal.join("0.old"))?;
        fs::write(&config, "new config")?;
        fs::write(&extra, "new sibling")?;
        fs::create_dir(&cold)?;
        fs::write(&pin, b"new-rev")?;
        fs::rename(&main, journal.join("4.old"))?;
        fs::write(&main, b"new-rev")?;
        fs::rename(&order, journal.join("5.old"))?;
        fs::write(&order, b"new order")?;
        drop(writer); // OS releases this lock when a publishing process exits.
        assert_eq!(
            get_cached_model_path_in(model, false, root.path().to_owned()).unwrap(),
            Some(snapshot)
        );
        assert_eq!(fs::read(config)?, b"{}");
        assert!(!extra.exists());
        assert!(!cold.exists());
        assert!(!pin.exists());
        assert_eq!(fs::read(order)?, b"original order");
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
    fn rollback_does_not_restore_an_evicted_previous_main() -> anyhow::Result<()> {
        let model = "org/model";
        for (cold, installed_order) in [(false, false), (false, true), (true, true)] {
            let root = TempDir::new()?;
            let next = build_hf_cache(
                root.path(),
                model,
                &["config.json", "tokenizer.json", "model.safetensors"],
            );
            let rev = next.file_name().unwrap().to_str().unwrap();
            let repo = next.parent().unwrap().parent().unwrap();
            let name = repo.file_name().unwrap();
            let main = repo.join("refs/main");
            let writer = lock_hf_publication(root.path(), name)?;
            let lock = hf_publication_lock_path(&root.path().join(".locks"), name);
            let journal = lock.with_extension("merge");
            let state = lock.with_extension("order");
            fs::create_dir(&journal)?;
            let update = if cold {
                next.clone()
            } else {
                next.join("config.json")
            };
            if !cold {
                fs::write(journal.join("0.old"), b"original B")?;
            }
            let manifest = [
                (update.strip_prefix(repo)?, !cold),
                (Path::new("refs/main"), true),
                (Path::new(HF_ORDER_FILE), true),
            ];
            fs::write(journal.join("manifest"), serde_json::to_vec(&manifest)?)?;
            let saved_order = |revision: &str, ticket| {
                serde_json::to_vec(&HfRequestOrder {
                    issued: 9,
                    main: Some(PublishedMain {
                        ticket,
                        revision: revision.to_owned(),
                        kind: PublishedSnapshotKind::Full,
                    }),
                    metadata_only: Some(["unrelated".to_owned()].into_iter().collect()),
                    ..Default::default()
                })
            };
            if installed_order {
                fs::write(journal.join("2.old"), saved_order("evicted", 3)?)?;
                fs::write(&state, saved_order(rev, 8)?)?;
            } else {
                fs::write(&state, saved_order("evicted", 3)?)?;
                fs::write(journal.join("2"), saved_order(rev, 8)?)?;
            }
            // Recovery must fail closed when it cannot inspect a ref backup.
            fs::create_dir(journal.join("1.old"))?;
            drop(writer);
            assert!(get_cached_model_path_in(model, false, root.path().into()).is_err());
            assert!(journal.is_dir());
            fs::remove_dir(journal.join("1.old"))?;
            fs::write(journal.join("1.old"), b"evicted")?;

            let found = get_cached_model_path_in(model, false, root.path().into())?;
            assert_eq!(found.as_deref(), (!cold).then_some(next.as_path()));
            let order = HfRequestOrder::read(&state)?;
            assert_eq!(order.issued, 9);
            assert_eq!(
                order.main.as_ref().map(|m| (m.revision.as_str(), m.ticket)),
                (!cold).then_some((rev, 8))
            );
            assert!(order.metadata_only.unwrap().contains("unrelated"));
            if cold {
                assert!(!main.exists());
            } else {
                assert_eq!(fs::read_to_string(main)?, rev);
                assert_eq!(fs::read(next.join("config.json"))?, b"original B");
            }
            assert!(!journal.exists());
        }
        Ok(())
    }

    #[test]
    fn revision_eviction_preserves_surviving_refs_and_publication_order() -> anyhow::Result<()> {
        for old_main_evicted in [true, false] {
            let root = TempDir::new()?;
            let model = "org/model";
            let name = OsStr::new("models--org--model");
            let repo = root.path().join(name);
            let kept = build_hf_cache(
                root.path(),
                model,
                &["config.json", "tokenizer.json", "model.safetensors"],
            );
            let kept_rev = kept.file_name().unwrap().to_str().unwrap();
            let removed = "removed";
            let refs = repo.join("refs");
            let old_main = if old_main_evicted { removed } else { kept_rev };
            let writer = lock_hf_publication(root.path(), name)?;
            let lock = hf_publication_lock_path(&root.path().join(".locks"), name);
            let journal = lock.with_extension("merge");
            let state = lock.with_extension("order");
            fs::create_dir(&journal)?;
            fs::write(
                journal.join("manifest"),
                serde_json::to_vec(&[
                    ("snapshots/removed/config.json", true),
                    ("refs/main", true),
                    ("refs/removed", true),
                    ("refs/new-pin", false),
                    (HF_ORDER_FILE, true),
                ])?,
            )?;
            fs::write(journal.join("0.old"), b"must not resurrect")?;
            fs::write(journal.join("1.old"), old_main)?;
            fs::write(journal.join("2.old"), removed)?;
            fs::write(
                journal.join("4.old"),
                serde_json::to_vec(&HfRequestOrder {
                    issued: 8,
                    main: Some(PublishedMain {
                        ticket: 6,
                        revision: old_main.to_owned(),
                        kind: PublishedSnapshotKind::Full,
                    }),
                    metadata_only: Some(
                        [removed.to_owned(), "untouched".to_owned()]
                            .into_iter()
                            .collect(),
                    ),
                    direct_models: [removed, "untouched"]
                        .into_iter()
                        .map(|revision| (revision.to_owned(), vec!["vocab.json".to_owned()]))
                        .collect(),
                })?,
            )?;
            if !old_main_evicted {
                // Eviction removed the interrupted replacement of refs/main.
                fs::remove_file(refs.join("main"))?;
            }
            // Also tolerate a ref that eviction did not finish unlinking.
            fs::write(refs.join(removed), removed)?;
            fs::write(refs.join("untouched"), kept_rev)?;
            drop(writer);

            let removed_path = repo.join("snapshots").join(removed);
            fs::write(&removed_path, b"not a snapshot directory")?;
            assert!(get_cached_model_path_in(model, false, root.path().into()).is_err());
            assert!(journal.is_dir());
            fs::remove_file(&removed_path)?;
            assert_eq!(
                get_cached_model_path_in(model, false, root.path().into())?,
                Some(kept.clone())
            );
            assert!(!journal.exists());
            assert!(!repo.join("snapshots").join(removed).exists());
            assert!(!refs.join(removed).exists());
            assert!(!refs.join("new-pin").exists());
            assert_eq!(fs::read_to_string(refs.join("main"))?, kept_rev);
            assert_eq!(fs::read_to_string(refs.join("untouched"))?, kept_rev);
            let order = HfRequestOrder::read(&state)?;
            assert_eq!(order.issued, 8);
            assert_eq!(
                order.main.as_ref().map(|main| main.ticket),
                (!old_main_evicted).then_some(6)
            );
            assert_eq!(
                order.metadata_only.unwrap(),
                ["untouched".to_owned()].into_iter().collect()
            );
            assert_eq!(
                order
                    .direct_models
                    .keys()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                ["untouched"]
            );
        }
        Ok(())
    }

    #[test]
    fn evicted_merge_recovers_before_lookup_or_new_publication() -> anyhow::Result<()> {
        let model = "org/model";
        let name = OsStr::new("models--org--model");
        for reader_first in [true, false] {
            let root = TempDir::new()?;
            let repo = root.path().join(name);
            let existing = build_hf_cache(root.path(), model, &["config.json"]);
            let stage = TempDir::new_in(root.path())?;
            let ticket = reserve_hf_request(root.path(), name, stage.path())?;
            // Another request is already in flight when the publisher is killed.
            let newer = reserve_hf_request(root.path(), name, stage.path())?;
            let writer = lock_hf_publication(root.path(), name)?;
            let lock = hf_publication_lock_path(&root.path().join(".locks"), name);
            let journal = lock.with_extension("merge");
            let state = lock.with_extension("order");
            let mut before = HfRequestOrder::read(&state)?;
            before.main = Some(PublishedMain {
                ticket,
                revision: existing.file_name().unwrap().to_str().unwrap().to_owned(),
                kind: PublishedSnapshotKind::Full,
            });
            fs::create_dir(&journal)?;
            let manifest = [
                (
                    existing.join("config.json").strip_prefix(&repo)?.to_owned(),
                    true,
                ),
                (PathBuf::from(HF_ORDER_FILE), true),
            ];
            fs::write(journal.join("manifest"), serde_json::to_vec(&manifest)?)?;
            fs::rename(existing.join("config.json"), journal.join("0.old"))?;
            fs::write(journal.join("1.old"), serde_json::to_vec(&before)?)?;
            fs::remove_file(&state)?; // Non-hardlink backup moved out of the live path.
            fs::remove_dir_all(&repo)?;
            drop(writer);

            if reader_first {
                fs::write(&repo, b"obstructed")?;
                assert!(get_cached_model_path_in(model, false, root.path().into()).is_err());
                assert!(journal.is_dir());
                fs::remove_file(&repo)?;
                assert!(get_cached_model_path_in(model, false, root.path().into())?.is_none());
                assert!(!repo.exists());
                let cleared = HfRequestOrder::read(&state)?;
                assert_eq!(cleared.issued, newer);
                assert!(cleared.main.is_none());
            }

            let snapshot = build_hf_cache(
                stage.path(),
                model,
                &["config.json", "tokenizer.json", "model.safetensors"],
            );
            let fresh = snapshot.with_file_name("new-revision");
            fs::rename(snapshot, &fresh)?;
            publish_hf_snapshot(
                stage.path(),
                &fresh,
                root.path(),
                None,
                FULL_MODEL,
                HfModelValidation::Stream,
                ticket,
            )?;
            assert_eq!(fs::read(repo.join("refs/main"))?, b"new-revision");
            assert!(!existing.exists());
            assert!(!journal.exists());
            assert!(reserve_hf_request(root.path(), name, stage.path())? > newer);
        }
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
    async fn offline_fetch_skips_provider_preferred_incomplete_snapshot() -> anyhow::Result<()> {
        let cache = TempDir::new()?;
        let model = "org/offline-partial";
        let files = ["config.json", "tokenizer.json", "model.safetensors"];
        let first = build_hf_cache(cache.path(), model, &files);
        let main = first.join("../../refs/main");
        let second = first.with_file_name("alternate");
        fs::create_dir(&second)?;
        for name in files {
            fs::write(second.join(name), "{}")?;
        }
        fs::remove_file(&main)?;
        let preferred =
            mx_cache::resolve_model_path(cache.path(), MxModelProvider::HuggingFace, model, None)?;
        let complete = if preferred == first {
            second
        } else {
            assert_eq!(preferred, second);
            first
        };
        fs::remove_file(preferred.join("model.safetensors"))?;
        temp_env::async_with_vars(
            [
                ("HF_HUB_CACHE", Some(cache.path().as_os_str())),
                ("HF_HUB_OFFLINE", Some("1".as_ref())),
                ("MODEL_EXPRESS_NO_SHARED_STORAGE", Some("1".as_ref())),
                ("MODEL_EXPRESS_URL", Some("http://[".as_ref())),
            ],
            async {
                assert_eq!(from_hf(model, false).await?, complete);
                assert!(!cache.path().join(".locks").exists());
                fs::write(&main, preferred.file_name().unwrap().as_encoded_bytes())?;
                assert_eq!(from_hf(model, false).await?, complete);
                assert!(!cache.path().join(".locks").exists());
                fs::write(preferred.join("model.safetensors"), "{}")?;
                fs::write(main, complete.file_name().unwrap().as_encoded_bytes())?;
                assert_eq!(from_hf(model, false).await?, complete);
                for path in [&complete, &preferred] {
                    fs::remove_file(path.join("model.safetensors"))?;
                }
                assert!(
                    from_hf(model, false)
                        .await
                        .unwrap_err()
                        .to_string()
                        .contains("incomplete")
                );
                anyhow::Ok(())
            },
        )
        .await
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn shared_offline_prefers_complete_snapshots_and_keeps_custom_layouts()
    -> anyhow::Result<()> {
        let cache = TempDir::new()?;
        let model = "org/shared-offline";
        let snapshot = build_hf_cache(cache.path(), model, &["config.json", "tokenizer.json"]);
        let alternate = snapshot.with_file_name("alternate");
        temp_env::async_with_vars(
            [
                ("HF_HUB_CACHE", Some(cache.path().as_os_str())),
                ("HF_HUB_OFFLINE", Some("1".as_ref())),
                ("MODEL_EXPRESS_NO_SHARED_STORAGE", None),
                ("MODEL_EXPRESS_URL", Some("http://[".as_ref())),
            ],
            async {
                fs::write(snapshot.join("custom.weights"), b"custom weights")?;
                assert_eq!(from_hf(model, false).await?, snapshot);
                fs::create_dir(&alternate)?;
                for name in ["config.json", "tokenizer.json", "model.safetensors"] {
                    fs::write(alternate.join(name), "{}")?;
                }
                assert_eq!(from_hf(model, false).await?, alternate);

                // Publish metadata at B while retaining full alternate as main.
                fs::write(snapshot.join("../../refs/main"), "alternate")?;
                let stage = TempDir::new_in(cache.path())?;
                let downloaded =
                    build_hf_cache(stage.path(), model, &["config.json", "tokenizer.json"]);
                let staged = downloaded.with_file_name("metadata-only");
                fs::rename(downloaded, &staged)?;
                let metadata = publish_streamed_hf_snapshot(
                    stage.path(),
                    &staged,
                    cache.path(),
                    Some("metadata-only"),
                    SnapshotRequest::Model {
                        ignore_weights: true,
                    },
                )?;
                assert_eq!(fs::read(snapshot.join("../../refs/main"))?, b"alternate");
                fs::remove_dir_all(alternate)?;
                assert_eq!(from_hf(model, false).await?, snapshot);
                fs::remove_dir_all(snapshot)?;
                assert!(
                    from_hf(model, false)
                        .await
                        .unwrap_err()
                        .to_string()
                        .contains("incomplete")
                );
                assert_eq!(from_hf(model, true).await?, metadata);

                // A later full publication at B clears its metadata-only marker.
                let full = TempDir::new_in(cache.path())?;
                let downloaded = build_hf_cache(
                    full.path(),
                    model,
                    &["config.json", "tokenizer.json", "custom.weights"],
                );
                let staged = downloaded.with_file_name("metadata-only");
                fs::rename(downloaded, &staged)?;
                publish_hf_snapshot(
                    full.path(),
                    &staged,
                    cache.path(),
                    Some("metadata-only"),
                    FULL_MODEL,
                    HfModelValidation::SharedProvider,
                    0,
                )?;
                assert_eq!(from_hf(model, false).await?, metadata);
                anyhow::Ok(())
            },
        )
        .await
    }

    #[test]
    fn direct_native_consolidated_checkout_allows_backend_tokenizer() -> anyhow::Result<()> {
        let cache = TempDir::new()?;
        let stage = TempDir::new_in(cache.path())?;
        let model = "org/native-delegated";
        let staged = build_hf_cache(
            stage.path(),
            model,
            &["params.json", "tokenizer.model", "consolidated.safetensors"],
        );
        assert!(
            publish_streamed_hf_snapshot(stage.path(), &staged, cache.path(), None, FULL_MODEL)
                .is_err()
        );
        let published = publish_hf_snapshot(
            stage.path(),
            &staged,
            cache.path(),
            None,
            FULL_MODEL,
            HfModelValidation::Direct,
            1,
        )?;
        assert_eq!(
            get_cached_model_path_in(model, false, cache.path().into())?,
            Some(published.clone())
        );
        fs::remove_file(published.join("consolidated.safetensors"))?;
        assert_eq!(
            get_cached_model_path_in(model, false, cache.path().into())?,
            None
        );
        Ok(())
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn direct_fallback_reuses_delegated_tokenizer_layout_but_requires_weights()
    -> anyhow::Result<()> {
        let root = TempDir::new()?;
        let cache = root.path().join("hub");
        let model = "org/delegated";
        let incomplete = "org/delegated-no-weights";
        let revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let mut hub = mockito::Server::new_async().await;
        let endpoint = hub.url();
        let mut mocks = Vec::new();
        for (name, complete) in [(model, true), (incomplete, false)] {
            let mut files = vec![
                ("config.json", r#"{"model_type":"gpt2"}"#),
                ("vocab.json", "{}"),
                ("merges.txt", "#version: 0.2\n"),
            ];
            if complete {
                files.push(("model.safetensors", "complete weights"));
            }
            let info = serde_json::json!({
                "id": name, "sha": revision,
                "siblings": files.iter().map(|(name, _)| serde_json::json!({"rfilename": name})).collect::<Vec<_>>(),
            });
            mocks.push(
                hub.mock("GET", format!("/api/models/{name}/revision/main").as_str())
                    .with_body(info.to_string())
                    .create_async()
                    .await,
            );
            for (file, body) in files {
                mocks.push(
                    hub.mock("GET", format!("/{name}/resolve/{revision}/{file}").as_str())
                        .with_header("x-repo-commit", revision)
                        .with_header("etag", file)
                        .with_body(body)
                        .expect_at_least(1)
                        .create_async()
                        .await,
                );
            }
        }
        let snapshot = temp_env::async_with_vars(
            [
                ("HOME", Some(root.path().as_os_str())),
                ("HF_HUB_CACHE", Some(cache.as_os_str())),
                ("HF_HUB_OFFLINE", None),
                ("HF_ENDPOINT", Some(endpoint.as_ref())),
                ("MODEL_EXPRESS_NO_SHARED_STORAGE", Some("1".as_ref())),
                ("MODEL_EXPRESS_URL", Some("http://[".as_ref())),
            ],
            async {
                let snapshot = from_hf(model, false).await?;
                assert!(!has_required_model_files(&snapshot, false)?);
                assert_eq!(from_hf(model, false).await?, snapshot);
                let error = from_hf(incomplete, false).await.unwrap_err();
                assert!(format!("{error:#}").contains("is incomplete"));
                assert!(!cache.join("models--org--delegated-no-weights").exists());

                let stage = TempDir::new_in(&cache)?;
                let streamed = build_hf_cache(
                    stage.path(),
                    model,
                    &[
                        "config.json",
                        "vocab.json",
                        "merges.txt",
                        "model.safetensors",
                    ],
                );
                assert!(
                    publish_streamed_hf_snapshot(stage.path(), &streamed, &cache, None, FULL_MODEL)
                        .is_err()
                );
                anyhow::Ok(snapshot)
            },
        )
        .await?;
        for mock in mocks {
            mock.assert_async().await;
        }

        fs::remove_file(snapshot.join("../../refs/main"))?;
        temp_env::async_with_vars(
            [
                ("HOME", Some(root.path().as_os_str())),
                ("HF_HUB_CACHE", Some(cache.as_os_str())),
                ("HF_HUB_OFFLINE", Some("1".as_ref())),
                ("HF_ENDPOINT", Some(endpoint.as_ref())),
                ("MODEL_EXPRESS_NO_SHARED_STORAGE", Some("1".as_ref())),
                ("MODEL_EXPRESS_URL", Some("http://[".as_ref())),
            ],
            async {
                assert_eq!(from_hf(model, false).await?, snapshot);
                fs::remove_file(snapshot.join("merges.txt"))?;
                assert!(from_hf(model, false).await.is_err());
                anyhow::Ok(())
            },
        )
        .await
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
        use sha2::{Digest, Sha256};
        let blobs = cache.join("models--org--partial/blobs");
        fs::create_dir_all(&blobs)?;
        let verified_native = "valid-native.safetensors";
        let pristine_blob = format!("{:x}", Sha256::digest(b"cached weights"));
        fs::write(blobs.join(&pristine_blob), b"cached weights")?;
        std::os::unix::fs::symlink(
            format!("../../blobs/{pristine_blob}"),
            partial.join(verified_native),
        )?;
        // git hash-object of the original {}, replaced here by equal-length [].
        let tokenizer_blob = "9e26dfeeb6e641a33dae4961196235bdb965b21b";
        fs::write(blobs.join(tokenizer_blob), b"[]")?;
        std::os::unix::fs::symlink(
            format!("../../blobs/{tokenizer_blob}"),
            partial.join("tokenizer.json"),
        )?;
        let valid_git = "special_tokens_map.json";
        let valid_blob = "a9d0d8f0794c094f630657629f7026a4d4126f83";
        fs::write(blobs.join(valid_blob), b"{\"valid\":true}")?;
        std::os::unix::fs::symlink(format!("../../blobs/{valid_blob}"), partial.join(valid_git))?;
        let unverified_git = "tokenizer_config.json";
        fs::write(blobs.join("unverified"), b"[]")?;
        std::os::unix::fs::symlink("../../blobs/unverified", partial.join(unverified_git))?;
        fs::write(partial.join("../../refs/main"), current)?;
        let verified = "model-00001-of-00002.safetensors";
        let wrong = "model-00002-of-00002.safetensors";
        fs::write(partial.join(verified), b"verified loose weights")?;
        fs::write(partial.join(wrong), b"spoiled")?;
        let short = "truncated-native.safetensors";
        std::os::unix::fs::symlink(format!("../../blobs/{pristine_blob}"), partial.join(short))?;
        let corrupt = "corrupt-native.safetensors";
        let corrupted_blob = format!("{:x}", Sha256::digest(b"good"));
        fs::write(blobs.join(&corrupted_blob), b"evil")?;
        std::os::unix::fs::symlink(
            format!("../../blobs/{corrupted_blob}"),
            partial.join(corrupt),
        )?;
        let nested = "nested/part.safetensors";
        let index = "model.safetensors.index.json";
        let indexed_weights = r#"{"weight_map":{"first":"./nested//part.safetensors","second":"model-00001-of-00002.safetensors"}}"#;
        fs::create_dir(partial.join("nested"))?;
        fs::write(partial.join(nested), b"nested weights")?;
        // Only the Hub has the index, and ModelExpress will skip the nested artifact.
        assert!(!partial.join(index).exists());
        let describe_lfs = |name, content: &[u8]| {
            serde_json::json!({"rfilename": name, "lfs": {
                "size": content.len(),
                "sha256": format!("{:x}", Sha256::digest(content))
            }})
        };
        let siblings = [
            serde_json::json!({"rfilename": "config.json"}),
            serde_json::json!({"rfilename": "tokenizer.json", "blobId": tokenizer_blob}),
            serde_json::json!({"rfilename": valid_git, "blobId": valid_blob}),
            serde_json::json!({"rfilename": unverified_git}),
            serde_json::json!({"rfilename": index}),
            describe_lfs(verified_native, b"cached weights"),
            describe_lfs(verified, b"verified loose weights"),
            describe_lfs(nested, b"nested weights"),
            describe_lfs(corrupt, b"good"),
            describe_lfs(wrong, b"trusted"),
            describe_lfs(short, b"complete native weights"),
        ];
        let info = serde_json::json!({"id":model, "sha":current, "siblings":siblings});
        let branch_lookup = hub
            .mock("GET", "/api/models/org/partial/revision/main")
            .match_query(mockito::Matcher::UrlEncoded("blobs".into(), "true".into()))
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
        let mut cached = Vec::new();
        for name in [verified_native, verified, nested, valid_git] {
            cached.push(
                hub.mock("GET", format!("/{model}/resolve/{current}/{name}").as_str())
                    .expect(0)
                    .create_async()
                    .await,
            );
        }
        let mut missing = Vec::new();
        for (name, body) in [
            ("config.json", "{}"),
            ("tokenizer.json", "{}"),
            (unverified_git, "{}"),
            (wrong, "trusted"),
            (short, "complete native weights"),
            (corrupt, "good"),
            (index, indexed_weights),
        ] {
            missing.push(
                hub.mock("GET", format!("/{model}/resolve/{current}/{name}").as_str())
                    .with_header("x-repo-commit", current)
                    .with_header("etag", name)
                    .with_body(body)
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
                let lock = hf_publication_lock_path(
                    &cache.join(".locks"),
                    OsStr::new("models--org--partial"),
                );
                let order = HfRequestOrder::read(&lock.with_extension("order"))?;
                assert_eq!(order.issued, 1, "stream and fallback belong to one fetch");
                assert_eq!(order.main.unwrap().ticket, 1);
                assert_eq!(fs::read(partial.join(verified_native))?, b"cached weights");
                assert_eq!(fs::read(partial.join("tokenizer.json"))?, b"{}");
                assert_eq!(fs::read(partial.join(valid_git))?, b"{\"valid\":true}");
                assert_eq!(fs::read(partial.join(unverified_git))?, b"{}");
                assert_eq!(fs::read(partial.join(wrong))?, b"trusted");
                assert_eq!(fs::read(partial.join(short))?, b"complete native weights");
                assert_eq!(fs::read(partial.join(corrupt))?, b"good");
                assert_eq!(fs::read(partial.join(nested))?, b"nested weights");
                assert_eq!(fs::read_to_string(partial.join(index))?, indexed_weights);
                assert!(has_required_model_files(&partial, false)?);
                branch_lookup.assert_async().await;
                commit_lookup.assert_async().await;
                for file in cached {
                    file.assert_async().await;
                }
                for file in missing {
                    file.assert_async().await;
                }
                anyhow::Ok(())
            },
        )
        .await
    }

    #[test]
    #[cfg(unix)]
    fn hf_seed_rejects_unsafe_or_unverifiable_nested_shards() -> anyhow::Result<()> {
        use sha2::{Digest, Sha256};

        let cache = TempDir::new()?;
        let repo = cache.path().join("models--org--nested");
        let old = build_hf_cache(cache.path(), "org/nested", &[]);
        let revision = old.file_name().unwrap().to_string_lossy();
        let stage = TempDir::new()?;
        let stage_repo = stage.path().join(repo.file_name().unwrap());
        let weights = old.join("weights");
        fs::create_dir(&weights)?;
        fs::write(weights.join("valid.safetensors"), b"data")?;
        fs::write(weights.join("wrong.safetensors"), b"evil")?;
        fs::write(weights.join("unverified.safetensors"), b"data")?;
        fs::write(weights.join("unindexed.safetensors"), b"data")?;
        let external = TempDir::new()?;
        fs::write(external.path().join("part.safetensors"), b"data")?;
        std::os::unix::fs::symlink(external.path(), old.join("alias"))?;
        let index = "model.safetensors.index.json";
        fs::write(
            old.join(index),
            r#"{"weight_map":{"old":"weights/unindexed.safetensors"}}"#,
        )?;
        let describe = |name: &str, verified: bool| HfSeedFile {
            rfilename: name.to_owned(),
            blob_id: None,
            lfs: verified.then(|| HfLfsFile {
                sha256: format!("{:x}", Sha256::digest(b"data")),
                size: 4,
            }),
        };
        let info = HfSeedManifest {
            sha: revision.into_owned(),
            siblings: vec![
                describe("weights/valid.safetensors", true),
                describe("weights/wrong.safetensors", true),
                describe("weights/unverified.safetensors", false),
                describe("weights/unindexed.safetensors", true),
                describe("alias/part.safetensors", true),
                describe("weights/../escape.safetensors", true),
                describe("/escape.safetensors", true),
            ],
        };
        seed_hf_checkout(&repo, &stage_repo, &info, &FULL_MODEL)?;
        let seeded = stage_repo.join("snapshots").join(&info.sha);
        assert!(!seeded.join("weights").exists());
        fs::write(
            seeded.join(index),
            r#"{"weight_map":{"valid":"./weights//valid.safetensors","wrong":"weights/wrong.safetensors","unverified":"weights/unverified.safetensors","alias":"alias/part.safetensors"}}"#,
        )?;
        recover_indexed_hf_shards(&repo, &seeded, &info)?;
        assert_eq!(fs::read(seeded.join("weights/valid.safetensors"))?, b"data");
        for name in [
            "wrong.safetensors",
            "unverified.safetensors",
            "unindexed.safetensors",
        ] {
            assert!(!seeded.join("weights").join(name).exists());
        }
        assert!(!seeded.join("alias").exists());
        assert!(normalized_hf_path("weights/../escape.safetensors").is_none());
        assert!(normalized_hf_path("/escape.safetensors").is_none());
        assert!(index_shards(br#"{"weight_map":{"a":"weights/../escape.safetensors"}}"#).is_none());
        Ok(())
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
