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
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "xnnpack.h"  // from @XNNPACK
#include "tflite/c/common.h"
#include "tflite/delegates/xnnpack/file_util.h"
#include "tflite/delegates/xnnpack/mmap_handle.h"
#include "tflite/delegates/xnnpack/weight_cache_schema_generated.h"
#include "tflite/delegates/xnnpack/weight_cache.h"


namespace tflite {
namespace xnnpack {

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

  bool DumpTensorIdentifierMapToFile(const std::string& dump_file_path) const ;
  std::string DumpTensorIdentifierMap() const;
  // In case a constant buffer data needs to be moved for some reason, this will
  // map the new buffer data to its identifier.
  void RemapDataBuffer(const void* buffer, const void* new_buffer);

  // Returns the offset of the buffer identified by `cache_key`.
  //
  // If the buffer isn't found, return SIZE_MAX.
  [[nodiscard]]
  size_t LookUp(const xnn_weights_cache_look_up_key* cache_key);

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

  // 캐시 구조를 사람이 읽을 수 있는 형태로 덤프
  std::string DumpWeightCacheStructure() const;
  
  // 덤프 결과를 파일로 저장
  bool DumpWeightCacheStructureToFile(const std::string& dump_file_path) const;
  
  // 간단한 통계 정보 반환
  std::string GetWeightCacheStats() const;

  void InitManagedBuffer(size_t size);

  void PrefetchFromFile(const std::string& filename);


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
      /*delete_cache=*/StreamingWeightCacheProvider::delete_cache};

  // Path to the cache file.
  std::string file_path_;

  // Maps buffer addresses to buffer identifiers.
  std::unordered_map<const void*, uint64_t> buffer_address_to_identifier_;

  std::unordered_map<const void*, const void*> buffer_remaps_;

  // Maps cache request hashes to the buffer identifier.
  std::unordered_multimap<PackIdentifier, BufferLocation, PackIdentifier::Hash>
      cache_key_to_offset_;

  // MMap allocation handler.
  std::vector<MMapHandle> mmap_handles_;

  // The offset to the first buffer data in the MMap allocation.
  size_t mmap_buffer_base_offset_;

  // Holds a file descriptor to the cache file.
  FileDescriptor file_descriptor_;

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

  //! READY FOR IMPLEMENT DOUBLE BUFFERING -> We need to modify this function to return the address from the active buffer
  void* managed_buffer_[2] = {nullptr, nullptr};
  size_t managed_size_ = 0;
  int active_buffer_index_ = 0;

  struct BufferEntry {
    void* buffers[2];   // double buffer 실제 주소
    int active;         // 현재 active 버퍼 인덱스
    size_t size;        // 버퍼 크기
  };
  
  std::unordered_map<size_t, BufferEntry> offset_to_entry_;
 
};


class ChunkManagementModule{
    //TODO: Chunk management logic 

};
class MemoryPrefetchModule{
    // Prefetching logic for memory managem
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_STREAMING_WEIGHT_CACHE_H_
