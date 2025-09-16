// Copyright 2025
// StreamingWeightCacheProvider implementation using composition pattern.
// Build operations delegate to MMapWeightCacheProvider, streaming operations
// use custom lightweight implementation.

#include "tflite/delegates/xnnpack/streaming_weight_cache.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

// FlatBuffers includes - TODO: fix include path issues
// #include "flatbuffers/verifier.h"  // from @flatbuffers
#include "tflite/delegates/xnnpack/file_util.h"
// #include "tflite/delegates/xnnpack/weight_cache_schema_generated.h"
#include "tflite/minimal_logging.h"

namespace tflite {
namespace xnnpack {

StreamingWeightCacheProvider::~StreamingWeightCacheProvider() {
    // Clean up materialized blobs
    for (auto& [offset, blob] : materialized_blobs_) {
        if (blob.data) {
            std::free(blob.data);
        }
    }
}

// === Build Operations (Delegate to MMapWeightCacheProvider) ===

void StreamingWeightCacheProvider::EnsureBuildProvider() {
    if (!build_provider_) {
        build_provider_ = std::make_unique<MMapWeightCacheProvider>();
    }
}

bool StreamingWeightCacheProvider::StartBuild(const char* file_path, FileDescriptor fd) {
    EnsureBuildProvider();
    return build_provider_->StartBuild(file_path, std::move(fd));
}

bool StreamingWeightCacheProvider::StartBuildStep() {
    if (!build_provider_) {
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR, 
                        "StreamingWeightCache: StartBuildStep called without StartBuild");
        return false;
    }
    return build_provider_->StartBuildStep();
}

bool StreamingWeightCacheProvider::StopBuildStep() {
    if (!build_provider_) {
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                        "StreamingWeightCache: StopBuildStep called without StartBuild");
        return false;
    }
    return build_provider_->StopBuildStep();
}

bool StreamingWeightCacheProvider::CanStartBuildStep() const {
    return build_provider_ && build_provider_->CanStartBuildStep();
}

size_t StreamingWeightCacheProvider::LookUp(const xnn_weights_cache_look_up_key* cache_key) {
    if (IsInBuildMode()) {
        return build_provider_->LookUp(cache_key);
    }
    // TODO: Implement streaming lookup using PackIdentifier
    return SIZE_MAX;
}

void* StreamingWeightCacheProvider::ReserveSpace(size_t size) {
    if (IsInBuildMode()) {
        return build_provider_->ReserveSpace(size);
    }
    // Streaming mode doesn't support reserving space
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "StreamingWeightCache: ReserveSpace not supported in streaming mode");
    return nullptr;
}

size_t StreamingWeightCacheProvider::LookUpOrInsert(const xnn_weights_cache_look_up_key* cache_key,
                                                   void* ptr, size_t size) {
    if (IsInBuildMode()) {
        return build_provider_->LookUpOrInsert(cache_key, ptr, size);
    }
    // Streaming mode is read-only, behaves like LookUp
    return LookUp(cache_key);
}

void StreamingWeightCacheProvider::SetFilePath(const char* file_path) {
    file_path_ = file_path;
    if (build_provider_) {
        build_provider_->SetFilePath(file_path);
    }
}

const std::string& StreamingWeightCacheProvider::GetFilePath() const {
    if (build_provider_) {
        return build_provider_->GetFilePath();
    }
    return file_path_;
}

void StreamingWeightCacheProvider::MapTensorIdentifiers(
    const TfLiteTensor* tensors, size_t size,
    const std::unordered_map<size_t, size_t>& tensor_index_to_identifier) {
    if (build_provider_) {
        build_provider_->MapTensorIdentifiers(tensors, size, tensor_index_to_identifier);
    }
    // TODO: Store for streaming mode if needed
}

void StreamingWeightCacheProvider::RemapDataBuffer(const void* buffer, const void* new_buffer) {
    if (build_provider_) {
        build_provider_->RemapDataBuffer(buffer, new_buffer);
    }
    // TODO: Handle remapping in streaming mode if needed
}

void StreamingWeightCacheProvider::Release() {
    if (build_provider_) {
        build_provider_->Release();
    }
    
    // Clean up streaming resources
    for (auto& [offset, blob] : materialized_blobs_) {
        if (blob.data) {
            std::free(blob.data);
        }
    }
    materialized_blobs_.clear();
    total_materialized_bytes_ = 0;
    
    if (streaming_fd_.IsValid()) {
        streaming_fd_.Close();
    }
}

bool StreamingWeightCacheProvider::IsBuilding() const {
    return build_provider_ && build_provider_->IsBuilding();
}

bool StreamingWeightCacheProvider::IsActive() const {
    return (build_provider_ && build_provider_->IsActive()) || IsInStreamingMode();
}

// === XNNPACK Provider Interface (Static Callbacks) ===

size_t StreamingWeightCacheProvider::look_up(void* context, 
                                            const xnn_weights_cache_look_up_key* cache_key) {
    auto* provider = static_cast<StreamingWeightCacheProvider*>(context);
    return provider->LookUp(cache_key);
}

void* StreamingWeightCacheProvider::reserve_space(void* context, size_t n) {
    auto* provider = static_cast<StreamingWeightCacheProvider*>(context);
    return provider->ReserveSpace(n);
}

size_t StreamingWeightCacheProvider::look_up_or_insert(
    void* context, const xnn_weights_cache_look_up_key* cache_key, void* ptr, size_t size) {
    auto* provider = static_cast<StreamingWeightCacheProvider*>(context);
    return provider->LookUpOrInsert(cache_key, ptr, size);
}

bool StreamingWeightCacheProvider::is_finalized(void* context) {
    auto* provider = static_cast<StreamingWeightCacheProvider*>(context);
    return !provider->IsBuilding();
}

void* StreamingWeightCacheProvider::offset_to_addr(void* context, size_t offset) {
    auto* provider = static_cast<StreamingWeightCacheProvider*>(context);
    return provider->OffsetToAddr(offset);
}

void StreamingWeightCacheProvider::delete_cache(void* context) {
    auto* provider = static_cast<StreamingWeightCacheProvider*>(context);
    delete provider;
}

// === Hybrid Address Resolution ===

void* StreamingWeightCacheProvider::OffsetToAddr(size_t offset) {
    if (IsInBuildMode()) {
        return build_provider_->OffsetToAddr(offset);
    }
    return StreamingOffsetToAddr(offset);
}

// === Debug/Instrumentation ===

size_t StreamingWeightCacheProvider::MaterializedMemoryBytes() const {
    return total_materialized_bytes_;
}

bool StreamingWeightCacheProvider::EvictBlob(size_t offset) {
    auto it = materialized_blobs_.find(offset);
    if (it != materialized_blobs_.end()) {
        if (it->second.data) {
            std::free(it->second.data);
            total_materialized_bytes_ -= it->second.size;
        }
        materialized_blobs_.erase(it);
        return true;
    }
    return false;
}

// === Streaming Implementation Stubs (TODO: Implement in next phase) ===

bool StreamingWeightCacheProvider::LoadOrStartBuild(const char* file_path, FileDescriptor fd) {
    // Route to appropriate mode based on file existence
    if (access(file_path, F_OK) == 0) {
        // File exists - use streaming mode
        return LoadStreamingIndex(file_path, std::move(fd));
    } else {
        // File doesn't exist - start build mode
        return StartBuild(file_path, std::move(fd));
    }
}

bool StreamingWeightCacheProvider::Load(const std::string& path, FileDescriptor fd) {
    return LoadStreamingIndex(path, std::move(fd));
}

bool StreamingWeightCacheProvider::Load() {
    if (file_path_.empty()) {
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                        "StreamingWeightCache: No file path set for Load()");
        return false;
    }
    return LoadStreamingIndex(file_path_, FileDescriptor());
}

bool StreamingWeightCacheProvider::LoadStreamingIndex(const std::string& path, FileDescriptor fd) {
    // TODO: Implement FlatBuffer index parsing
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                    "StreamingWeightCache: LoadStreamingIndex not yet implemented");
    return false;
}

bool StreamingWeightCacheProvider::ParseCacheHeader(XNNPackCacheHeader& header) {
    // TODO: Implement header parsing
    return false;
}

bool StreamingWeightCacheProvider::ParseFlatBufferIndex(const XNNPackCacheHeader& header) {
    // TODO: Implement FlatBuffer index parsing
    return false;
}

bool StreamingWeightCacheProvider::MaterializeBlob(size_t offset) {
    // TODO: Implement on-demand blob materialization via pread
    return false;
}

void* StreamingWeightCacheProvider::StreamingOffsetToAddr(size_t offset) {
    // Check if already materialized
    auto it = materialized_blobs_.find(offset);
    if (it != materialized_blobs_.end()) {
        return it->second.data;
    }
    
    // Materialize on demand
    if (MaterializeBlob(offset)) {
        auto it = materialized_blobs_.find(offset);
        if (it != materialized_blobs_.end()) {
            return it->second.data;
        }
    }
    
    return nullptr;
}

}  // namespace xnnpack
}  // namespace tflite