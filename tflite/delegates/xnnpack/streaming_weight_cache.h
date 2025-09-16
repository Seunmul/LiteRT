// Copyright 2025
// Streaming Weight Cache Provider - uses composition to combine MMapWeightCacheProvider's
// build functionality with custom streaming load functionality.
//
// Design Philosophy:
//  - Build mode: Delegate to internal MMapWeightCacheProvider (proven, stable)
//  - Load mode: Use lightweight streaming logic with minimal dependencies
//  - Minimal surface area: Only create MMapWeightCacheProvider when building
//
// Key Benefits of Composition:
//  - Zero modification to existing MMapWeightCacheProvider code
//  - Avoid inheriting unnecessary mmap-related state (mmap_handles_, etc.)
//  - Clear separation: build logic vs streaming logic
//  - Can switch strategies at runtime based on mode
//
// Phase 1 Goals:
//  - 100% compatible with existing cache file format and build logic
//  - On-demand blob materialization to reduce memory footprint
//  - Synchronous pread-based loading (no async I/O yet)
//  - Simple materialized blob cache (no LRU, no eviction)
//
// Future Phases:
//  - Phase 2: LRU cache with memory pressure handling
//  - Phase 3: Background prefetch and double buffering
//  - Phase 4: Async io_uring backend
//  - Phase 5: Compression and remote backends

#ifndef TFLITE_DELEGATES_XNNPACK_STREAMING_WEIGHT_CACHE_H_
#define TFLITE_DELEGATES_XNNPACK_STREAMING_WEIGHT_CACHE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "tflite/delegates/xnnpack/weight_cache.h"  // For MMapWeightCacheProvider

namespace tflite {
namespace xnnpack {

// Simple POD describing a materialized in-memory blob loaded on-demand.
struct MaterializedBlob {
	void* data = nullptr;      // owned heap allocation (aligned)
	size_t size = 0;           // size in bytes
};

// StreamingWeightCacheProvider uses composition to combine build and streaming logic.
// - Build operations: delegate to internal MMapWeightCacheProvider
// - Load operations: use custom streaming implementation
class StreamingWeightCacheProvider {
 public:
	StreamingWeightCacheProvider() = default;
	~StreamingWeightCacheProvider();

	// Non-copyable
	StreamingWeightCacheProvider(const StreamingWeightCacheProvider&) = delete;
	StreamingWeightCacheProvider& operator=(const StreamingWeightCacheProvider&) = delete;

	// Moveable
	StreamingWeightCacheProvider(StreamingWeightCacheProvider&&) = default;
	StreamingWeightCacheProvider& operator=(StreamingWeightCacheProvider&&) = default;
	
	// === Public Interface (mirrors MMapWeightCacheProvider) ===
	
	// Load existing cache OR start build - routing logic
	bool LoadOrStartBuild(const char* file_path, FileDescriptor fd = FileDescriptor());
	
	// Load existing cache - streaming implementation
	bool Load(const std::string& path, FileDescriptor fd = FileDescriptor());
	bool Load();
	
	// Build operations - delegate to internal MMapWeightCacheProvider
	bool StartBuild(const char* file_path, FileDescriptor fd = FileDescriptor());
	bool StartBuildStep();
	bool StopBuildStep();
	bool CanStartBuildStep() const;
	
	// Cache operations - hybrid implementation
	size_t LookUp(const xnn_weights_cache_look_up_key* cache_key);
	void* ReserveSpace(size_t size);
	size_t LookUpOrInsert(const xnn_weights_cache_look_up_key* cache_key, void* ptr, size_t size);
	
	// Address resolution - streaming or build delegate
	void* OffsetToAddr(size_t offset);
	
	// Utility operations
	void SetFilePath(const char* file_path);
	const std::string& GetFilePath() const;
	void MapTensorIdentifiers(const TfLiteTensor* tensors, size_t size,
	                         const std::unordered_map<size_t, size_t>& tensor_index_to_identifier);
	void RemapDataBuffer(const void* buffer, const void* new_buffer);
	void Release();
	bool IsBuilding() const;
	bool IsActive() const;
	
	// XNNPACK provider interface (static callbacks)
	static size_t look_up(void* context, const xnn_weights_cache_look_up_key* cache_key);
	static void* reserve_space(void* context, size_t n);
	static size_t look_up_or_insert(void* context, const xnn_weights_cache_look_up_key* cache_key, void* ptr, size_t size);
	static bool is_finalized(void* context);
	static void* offset_to_addr(void* context, size_t offset);
	static void delete_cache(void* context);
	
	// Get XNNPACK provider struct
	xnn_weights_cache_provider* GetCacheProvider() { return &cache_provider_; }
	
	// === Streaming-specific Debug/Instrumentation ===
	
	size_t MaterializedBlobCount() const { return materialized_blobs_.size(); }
	size_t MaterializedMemoryBytes() const { return total_materialized_bytes_; }
	bool EvictBlob(size_t offset);

 private:
	// === Mode Detection & Routing ===
	
	bool IsInBuildMode() const { return build_provider_ != nullptr; }
	bool IsInStreamingMode() const { return streaming_fd_.IsValid(); }
	void EnsureBuildProvider();
	
	// === Streaming Implementation ===
	
	bool LoadStreamingIndex(const std::string& path, FileDescriptor fd);
	bool ParseCacheHeader(XNNPackCacheHeader& header);
	bool ParseFlatBufferIndex(const XNNPackCacheHeader& header);
	bool MaterializeBlob(size_t offset);
	void* StreamingOffsetToAddr(size_t offset);
	
	// === State Management ===
	
	// Build mode: delegate to MMapWeightCacheProvider when building
	std::unique_ptr<MMapWeightCacheProvider> build_provider_;
	
	// Streaming mode: lightweight file access and blob cache
	FileDescriptor streaming_fd_;
	std::string file_path_;
	
	// Parsed index: logical offset -> file location
	std::unordered_map<size_t, BufferLocation> stream_index_;
	
	// Materialized blobs: logical offset -> heap allocation  
	std::unordered_map<size_t, MaterializedBlob> materialized_blobs_;
	size_t total_materialized_bytes_ = 0;
	
	// Cache file metadata (from header)
	uint64_t stream_base_offset_ = 0;  // FlatBuffer base_offset for address calculation
	
	// XNNPACK provider interface
	xnn_weights_cache_provider cache_provider_{
		this,
		&StreamingWeightCacheProvider::look_up,
		&StreamingWeightCacheProvider::reserve_space,
		&StreamingWeightCacheProvider::look_up_or_insert,
		&StreamingWeightCacheProvider::is_finalized,
		&StreamingWeightCacheProvider::offset_to_addr,
		&StreamingWeightCacheProvider::delete_cache
	};
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TFLITE_DELEGATES_XNNPACK_STREAMING_WEIGHT_CACHE_H_