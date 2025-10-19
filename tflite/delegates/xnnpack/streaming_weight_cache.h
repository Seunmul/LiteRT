/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_STREAMING_WEIGHT_CACHE_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_STREAMING_WEIGHT_CACHE_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "xnnpack.h"  // from @XNNPACK
#include "tflite/c/common.h"
#include "tflite/delegates/xnnpack/file_util.h"
#include "tflite/delegates/xnnpack/mmap_handle.h"
#include "tflite/delegates/xnnpack/weight_cache_schema_generated.h"
#include "tflite/delegates/xnnpack/weight_cache.h"


namespace tflite {
namespace xnnpack {

// Forward declaration
class StreamingWeightCacheProvider;
class WeightChunkPrefetcher;
class WeightChunkInfoWriter;
class PrefetchPlanLoader;

// Allows XNNPack to directly load packed weights from disk instead of having to
// repack them every time.
//
// XNNPack kernels do not have knowledge of the TFLite context. The only thing
// they can access is the buffers address. We rely on the fact that the address
// provided by TFLite is unique in order to find out the buffer identifier.
//
// To use the cache you need to:
//
//  - Map the buffer addresses to their identifier with `MapTensorIdentifiers`
//  - Load the cache file.
//  - Finalize the cache before calling the run functions of XNNPack (setup and
//    reshape are ok).
class StreamingWeightCacheProvider {
 public:
  StreamingWeightCacheProvider() = default;
  StreamingWeightCacheProvider(const StreamingWeightCacheProvider&) = delete;
  StreamingWeightCacheProvider& operator=(const StreamingWeightCacheProvider&) = delete;
  StreamingWeightCacheProvider(StreamingWeightCacheProvider&&);
  StreamingWeightCacheProvider& operator=(StreamingWeightCacheProvider&&);

  struct weight_chunk_info_t {
    size_t chunk_index;
    size_t aligned_offset;
    size_t offset_adjust; //abs_offset(mmap_buffer_base_offset_ + offset) - aligned_offset
    size_t aligned_size;
    size_t origin_offset;
    size_t origin_size;
    int managed_buffer_index; // which managed buffer was used to load this chunk
    size_t weights_id;
  };

  enum class ProviderMode {
    PRE_RUNTIME,  // record pre-runtime I/O prefetch plan
    RUNTIME,      // runtime streaming mode (default)
    DEBUG_MMAP    // mmap only for debugging
  };

  
  // WeightChunkPrefetcher가 private 멤버에 접근할 수 있도록 friend 선언
  friend class WeightChunkPrefetcher;
  friend class WeightChunkInfoWriter;

  // Changes the file path to save the cache to.
  //
  // WARNING: Can only be called if the cache isn't finalized.
  void SetFilePath(const char* file_path);

  const std::string& GetFilePath() const { return file_path_; }

  // Tries to load the given file. If the file doesn't exist starts building the
  // cache for it.
  //
  // If `fd` is provided, use that instead of reopening the file at the given
  // path.
  [[nodiscard /*Loading a cache file may fail.*/]]
  bool LoadOrStartBuild(const char* file_path,
                        FileDescriptor fd = FileDescriptor());

  [[nodiscard /*Starting to build a cache file may fail.*/]]
  bool StartBuild(const char* file_path, FileDescriptor fd = FileDescriptor());

  // Sets the weight file path and loads it.
  [[nodiscard /*Loading a cache file may fail.*/]]
  bool Load(const std::string& path, FileDescriptor fd = FileDescriptor());

  // Loads the weight cache previously set with `SetFilePath`.
  [[nodiscard /*Loading cache data may fail.*/]]
  bool Load();

  // Checks if the cache is currently being built or if it was loaded from a
  // file.
  [[nodiscard]]
  bool CanStartBuildStep() const {
    return building_run_;
  };

  // Prepares to add new data to the cache.
  [[nodiscard /*Updating cache data may fail.*/]]
  bool StartBuildStep();

  // Prepares to use data that was added to the cache during a build step.
  [[nodiscard /*Updating cache data may fail.*/]]
  bool StopBuildStep();

  // Creates the tensor map.
  void MapTensorIdentifiers(
      const TfLiteTensor* tensors, size_t size,
      const std::unordered_map<size_t, size_t>& tensor_index_to_identifier);

  // In case a constant buffer data needs to be moved for some reason, this will
  // map the new buffer data to its identifier.
  void RemapDataBuffer(const void* buffer, const void* new_buffer);

  // Returns the offset of the buffer identified by `cache_key`.
  //
  // If the buffer isn't found, return SIZE_MAX.
  [[nodiscard]]
  size_t LookUp(const xnn_weights_cache_look_up_key* cache_key);

  // Lookup by explicit identifiers (pack algorithm id, weights id, optional bias id).
  // This bypasses pointer->identifier mapping and directly queries the loaded
  // cache_key_to_offset_ map. Returns SIZE_MAX if not found.
  [[nodiscard]]
  size_t LookUpByIds(size_t pack_algorithm_id, size_t weights_id,
                     size_t bias_id = PackIdentifier::kNoId);

  // Reserves space for a buffer of given size and returns a pointer to it.
  //
  // The buffer data should be filled and `LookUpOrInsert` should be immediately
  // called.
  [[nodiscard]]
  void* ReserveSpace(size_t size);

  // Returns the offset of the buffer identified by `cache_key`. If the lookup
  // fails, inserts the span `[ptr, ptr+size)`.
  //
  // This should be called after ReserveSpace and `ptr` should be the result of
  // that call with the given `size`.
  //
  // WARNING: The cache key cannot be null.
  [[nodiscard]]
  size_t LookUpOrInsert(const xnn_weights_cache_look_up_key* cache_key,
                        void* ptr, size_t size);

  // Gets the pointer to the buffer at the given offset.
  //
  // WARNING: This requires the buffer to be finalized.
  // WARNING: This does not check the validity of the passed offset.
  void* OffsetToAddr(size_t offset);

  void PreInvokeHook(const size_t offset);

  void PostInvokeHook(const size_t offset);

  void TraceWeightsAddr(void* addr, const size_t offset);

  // Releases the weight cache's memory.
  void Release();

  // Returns true if any weights have been added to the underlying builder.
  [[nodiscard]]
  bool IsBuilding() const {
    return is_build_step_;
  };

  // Returns true if a file is mapped or a file path is set.
  [[nodiscard]]
  bool IsActive() const {
    return !mmap_handles_.empty() || builder_.IsStarted();
  };

  // Returns the cache provider expected by XNNPack.
  xnn_weights_cache_provider& GetCacheProvider() { return cache_provider_; }
  
  bool OpenDirectIOFileDescriptor(std::string file_path);
  bool CloseDirectIOFileDescriptor();

  // Initializes the double managed buffer with the given size.
  void AllocManagedBuffer(size_t size);
  
  void FreeManagedBuffer();
  
  void SwitchActiveBuffer();

  void ResetActiveBuffer();


  // WeightChunkPrefetcher Helpers
  void InitWeightChunkPrefetcher();

  WeightChunkPrefetcher* GetWeightChunkPrefetcher(){
    return weight_chunk_prefetcher_.get();
  }

  // WeightChunkInfoWriter Helpers
  void SetWeightChunkInfoWriter(WeightChunkInfoWriter* writer) {
        chunk_info_writer = writer;
  }


  /********* Utilities *********/

  // Returns the address that was recorded from the mmap mapping for the given
  // offset (i.e. the address stored in `offset_to_addr_`). This is useful to
  // compare the original mmap() address vs the actual address used at runtime
  // (which may be a managed buffer override). Returns nullptr if no mmaped
  // address is recorded for `offset`.
  void* GetMmappedAddr(size_t offset);

  // Returns a snapshot of the loaded cache key mappings. Each element is a
  // pair of PackIdentifier and BufferLocation reflecting what was loaded from
  // the on-disk FlatBuffer. This allows tools to inspect the exact
  // identifier→offset mappings without parsing the human-readable dump.
  std::unordered_multimap<PackIdentifier, BufferLocation, PackIdentifier::Hash> GetCacheKeyToOffset() const {
    return cache_key_to_offset_;
  }

  // Returns a copy of the internal buffer address -> identifier map.
  // Use this to take a stable snapshot for diagnostics or validation tools.
  std::unordered_map<const void*, uint64_t> GetBufferAddressToIdentifier() const {
    return buffer_address_to_identifier_;
  }

  
  std::string DumpTensorIdentifierMap() const;
  
  bool DumpTensorIdentifierMapToFile(const std::string& dump_file_path) const;
  
  std::string DumpWeightCacheStructure() const;
  
  bool DumpWeightCacheStructureToFile(const std::string& dump_file_path) const;
  
  bool VerifyBuffer(size_t offset);

  bool VerifyAllBuffers();


  void SetProviderMode(ProviderMode mode) { mode_ = mode; }
  ProviderMode GetProviderMode() const { return mode_; }
  std::string GetProviderModeString() const {
    switch (mode_) {
      case ProviderMode::PRE_RUNTIME:
        return "PRE_RUNTIME";
      case ProviderMode::RUNTIME:
        return "RUNTIME";
      case ProviderMode::DEBUG_MMAP:
        return "DEBUG_MMAP";
      default:
        return "UNKNOWN";
    }
  }


  /************ C Interfaces ************/

  // C interface: `xnn_weights_cache_provider` callback.
  static size_t look_up(void* context,
                        const xnn_weights_cache_look_up_key* cache_key);

  // C interface: `xnn_weights_cache_provider` callback.
  static void* reserve_space(void* context, size_t n);

  // C interface: `xnn_weights_cache_provider` callback.
  static size_t look_up_or_insert(
      void* context, const xnn_weights_cache_look_up_key* cache_key, void* ptr,
      size_t size);

  // C interface: `xnn_weights_cache_provider` callback.
  static bool is_finalized(void* context);

  // C interface: `xnn_weights_cache_provider` callback.
  static void* offset_to_addr(void* context, size_t offset);

  // C interface: `xnn_weights_cache_provider` callback.
  static enum xnn_status delete_cache(void* context);

  // C interface: `xnn_weights_cache_provider` callback.
  static void pre_invoke_hook(void* context, size_t offset);

  // C interface: `xnn_weights_cache_provider` callback.
  static void post_invoke_hook(void* context, size_t offset);

  // C interface: `xnn_weights_cache_provider` callback.
  static void trace_weights_addr(void* context, void* addr, size_t offset);
 private:
  // Hashes a cache key to lookup in `cache_key_to_identifier_`.
  PackIdentifier BuildPackIdentifier(const xnn_weights_cache_look_up_key& key);

  // Loads the data written by the last call to `builder_.BuildStepStop()`.
  [[nodiscard /*Loading cache data may fail.*/]]
  bool LoadLastBuildStep();

  // Cache provider implementation for XNNPack.
  xnn_weights_cache_provider cache_provider_{
      /*context=*/this,
      /*look_up=*/StreamingWeightCacheProvider::look_up,
      /*reserve_space=*/StreamingWeightCacheProvider::reserve_space,
      /*look_up_or_insert=*/StreamingWeightCacheProvider::look_up_or_insert,
      /*is_finalized=*/StreamingWeightCacheProvider::is_finalized,
      /*offset_to_addr=*/StreamingWeightCacheProvider::offset_to_addr,
      /*delete_cache=*/StreamingWeightCacheProvider::delete_cache,
#ifdef USE_WEIGHT_STREAMING
      /*pre_invoke_hook=*/StreamingWeightCacheProvider::pre_invoke_hook,
      /*post_invoke_hook=*/StreamingWeightCacheProvider::post_invoke_hook,
      /*trace_weights_addr=*/StreamingWeightCacheProvider::trace_weights_addr,
#endif
    };

  // Path to the cache file.
  std::string file_path_;

  // Maps buffer addresses to buffer identifiers.
  std::unordered_map<const void*, uint64_t> buffer_address_to_identifier_;

  std::unordered_map<const void*, const void*> buffer_remaps_;

  // Maps cache request hashes to the buffer identifier.
  std::unordered_multimap<PackIdentifier, BufferLocation, PackIdentifier::Hash>
      cache_key_to_offset_;

  // MMap handles to the file that contains the cache.
  std::vector<MMapHandle> mmap_handles_;
  // The base offset in the file where the weight data is stored.
  size_t mmap_buffer_base_offset_ = 0;
  // File descriptor for the cache file.
  FileDescriptor file_descriptor_;
  // File descriptor for direct I/O operations.
  FileDescriptor direct_io_file_descriptor_;
  // Used to build the cache.
  WeightCacheBuilder builder_;

  // True if the current run is the one building the cache file.
  //
  // We cannot distinguish between a wrong/outdated cache and one that is not
  // fully done. To detect misuse, we still want to raise an error when XNNPack
  // tries to append data to an existing file (i.e. when this is `false`).
  bool building_run_ = false;

  // True between StartBuildStep and StopBuildStep.
  //
  // This is used to check whether the builder is active, which means that some
  // of the buffers are not available/can't be retrieved.
  bool is_build_step_ = false;

  // Stores the loaded buffer addresses corresponding to the given offset in the
  // cache file.
  std::map<size_t, void*> offset_to_addr_;
  
  // Stores the size of each buffer stored at the given offset in the cache file.
  std::map<size_t, size_t> offset_to_size_;

  // Stores the weights id corresponding to the given offset in the cache file.
  std::map<size_t, int> offset_to_weights_id_;


  ProviderMode mode_ = ProviderMode::RUNTIME; // default: streaming at runtime

  //! READY FOR IMPLEMENT DOUBLE BUFFERING -> We need to modify this function to return the address from the active buffer
  void* managed_buffer_[2] = {nullptr, nullptr};
  size_t managed_buffer_size_ = 0;
  int active_buffer_index_ = 0;
  const size_t managed_buffer_sector_size_ = 4096; // direct I/O를 위한 섹터 크기

  std::unordered_map<size_t, weight_chunk_info_t> offset_to_weight_chunk_info_;
  std::unordered_map<size_t, weight_chunk_info_t> index_to_weight_chunk_info_;
    
  // Prefetcher 인스턴스
  std::unique_ptr<WeightChunkPrefetcher> weight_chunk_prefetcher_;
  
  // Stores the ptr address of the loaded buffer of weights corresponding to the given offset
  // each array element: [0] -> PREFETCHMODE: PREFILL, [1] -> PREFETCHMODE: DECODE
  std::map<size_t, std::array<void*, 2>> offset_to_weights_addr_ptr_;


  // WeightChunkInfoWriter Pointer (Dependency Injection)
  WeightChunkInfoWriter* chunk_info_writer = nullptr;
  
};

//* ============ WeightChunkPrefetcher ============ */


class WeightChunkPrefetcher{
public:
  WeightChunkPrefetcher() = default;
  ~WeightChunkPrefetcher() = default;

  enum class PrefetchState {
    IDLE,
    PREFETCHING,
    COMPLETED
  };

  enum class PrefetchMode {
    PREFILL,
    DECODE,
    UNINITIALIZED,
  };

  static inline int ModeToIndex(PrefetchMode mode) {
    switch (mode) {
      case PrefetchMode::PREFILL: return 0;
      case PrefetchMode::DECODE:  return 1;
      default: return -1;
    }
  }

  void Init(StreamingWeightCacheProvider* weight_cache_provider);
  
  struct PrefetchPlan {
    std::unordered_map<size_t, size_t> offset_to_index; // origin_offset -> index
    std::vector<StreamingWeightCacheProvider::weight_chunk_info_t> chunks;   // index -> info //TODO: Will be revmoved
  };

  // 모드별 프리패치 플랜 설정(소유권 이전)
  void SetPrefetchPlan(PrefetchMode mode,
                       std::unordered_map<size_t, size_t>&& offset_to_index,
                       std::vector<StreamingWeightCacheProvider::weight_chunk_info_t>&& chunks);

  // 플랜 유무 확인/조회(옵션)
  bool HasPlan(PrefetchMode mode) const;
  const PrefetchPlan* GetPlan(PrefetchMode mode) const;
  PrefetchPlan* GetPlan(PrefetchMode mode) = delete;
  
  bool LoadWeightChunk(size_t offset);

  // Build a unified index_to_chunks_ map by traversing all plans' chunks.
  // Key: chunk_index, Value: weight_chunk_info_t (full chunk metadata)
  // If the same chunk_index appears with a different metadata across modes,
  // the first occurrence is kept and a warning is logged (showing origin_offset).
  void BuildIndexToChunksFromPlans();

  // Accessor for the unified chunk table (index -> full chunk metadata).
  // Empty / unused indices will have chunk_index == SIZE_MAX.
  const std::vector<StreamingWeightCacheProvider::weight_chunk_info_t>& GetIndexToChunks() const { return index_to_chunks_; }
  
  void UpdatePrefetcherMode(PrefetchMode mode) { prefetch_mode_ = mode; }
  PrefetchMode GetPrefetcherMode() const { return prefetch_mode_; }
  std::string GetPrefetcherModeString() const {
    switch (prefetch_mode_) {
      case PrefetchMode::PREFILL:
        return "PREFILL";
      case PrefetchMode::DECODE:
        return "DECODE";
      case PrefetchMode::UNINITIALIZED:
        return "UNINITIALIZED";
      default:
        return "UNKNOWN";
    }
  }


private:

  

  StreamingWeightCacheProvider* weight_cache_provider_;
  PrefetchMode prefetch_mode_ = PrefetchMode::UNINITIALIZED;
  std::array<PrefetchPlan, 2> prefetch_plans_{}; // [PREFILL=0, DECODE=1]
  bool has_plan_[2] = {false, false};
  std::vector<StreamingWeightCacheProvider::weight_chunk_info_t>
      index_to_chunks_;
};

//* ============ WeightChunkInfoWriter ============ */

class WeightChunkInfoWriter {
public:
    virtual ~WeightChunkInfoWriter() = default;
    virtual void WriteChunkInfo(const StreamingWeightCacheProvider::weight_chunk_info_t& chunk_info,
                                WeightChunkPrefetcher::PrefetchMode prefetch_mode) = 0;
    virtual void Finalize() = 0;
};

//* ============ PrefetchPlanLoader ============ */

class PrefetchPlanLoader {
public:
    virtual ~PrefetchPlanLoader() = default;

    // Load prefetch plan from external source (e.g., JSON file)
    virtual bool LoadFromFile(const std::string& plan_file_path) = 0;
    
    
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_STREAMING_WEIGHT_CACHE_H_
