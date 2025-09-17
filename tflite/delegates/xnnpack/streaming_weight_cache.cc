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
#include "tflite/delegates/xnnpack/weight_cache.h"

#include <fcntl.h>
#if defined(_MSC_VER)
#include <io.h>
#define F_OK 0
#else
#include <unistd.h>
#endif

#include <cerrno>  // IWYU pragma: keep
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "xnnpack.h"  // from @XNNPACK
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "tflite/c/common.h"
#include "tflite/delegates/xnnpack/file_util.h"
#include "tflite/delegates/xnnpack/mmap_handle.h"
#include "tflite/delegates/xnnpack/weight_cache_schema_generated.h"
#include "tflite/logger.h"
#include "tflite/minimal_logging.h"

#include "tflite/delegates/xnnpack/weight_cache.h"
#include "tflite/delegates/xnnpack/streaming_weight_cache.h"

#define XNNPACK_ABORT_CHECK(TEST, ...)                      \
  if (!(TEST)) {                                            \
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR, __VA_ARGS__); \
    std::abort();                                           \
  }

#define XNNPACK_VAR_ARG_HEAD(FIRST, ...) FIRST

#define XNNPACK_RETURN_CHECK(TEST, ...)                              \
  if (!(TEST)) {                                                     \
    if (sizeof(XNNPACK_VAR_ARG_HEAD("" __VA_ARGS__)) > sizeof("")) { \
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,                      \
                      "XNNPack weight cache: " __VA_ARGS__);         \
    }                                                                \
    return false;                                                    \
  }

namespace tflite::xnnpack {

namespace {
constexpr size_t kMinAlignment = 128;

const char* Sanitize(const char* path) { return path ? path : ""; }

// Checks if the given path is a special value to use an in-memory cache.
bool IsInMemoryCachePath(const char* path) {
  // Use strncmp to check for the prefix.
  return path &&
         !strncmp(path, kInMemoryCachePath, sizeof(kInMemoryCachePath) - 1);
}

// Checks if the given path is a special value to use an in-memory cache.
bool IsInMemoryCachePath(const std::string& path) {
  // Use strncmp to check for the prefix.
  return IsInMemoryCachePath(path.c_str());
}

// Returns the next offset value that is aligned to `alignement`.
size_t Align(size_t offset, const size_t alignment) {
  const size_t misalign = offset % alignment;
  return offset + (misalign ? alignment - misalign : 0);
}

// Returns true if the given path exists.
[[nodiscard]]
bool FileExists(const char* path) {
  return access(path, F_OK) != -1;
}

}  // namespace

#define XNN_MOVE_CONSTRUCT_MEMBER(member) member(std::move(other.member))
StreamingWeightCacheProvider::StreamingWeightCacheProvider(
    StreamingWeightCacheProvider&& other)
    : XNN_MOVE_CONSTRUCT_MEMBER(cache_provider_),
      XNN_MOVE_CONSTRUCT_MEMBER(file_path_),
      XNN_MOVE_CONSTRUCT_MEMBER(buffer_address_to_identifier_),
      XNN_MOVE_CONSTRUCT_MEMBER(buffer_remaps_),
      XNN_MOVE_CONSTRUCT_MEMBER(cache_key_to_offset_),
      XNN_MOVE_CONSTRUCT_MEMBER(mmap_handles_),
      XNN_MOVE_CONSTRUCT_MEMBER(mmap_buffer_base_offset_),
      XNN_MOVE_CONSTRUCT_MEMBER(file_descriptor_),
      XNN_MOVE_CONSTRUCT_MEMBER(builder_),
      XNN_MOVE_CONSTRUCT_MEMBER(building_run_),
      XNN_MOVE_CONSTRUCT_MEMBER(is_build_step_),
      XNN_MOVE_CONSTRUCT_MEMBER(offset_to_addr_) {
  // The contexts need to keep pointing to their owning object.
  cache_provider_.context = this;
  other.cache_provider_.context = &other;
}
#undef XNN_MOVE_CONSTRUCT_MEMBER

StreamingWeightCacheProvider& StreamingWeightCacheProvider::operator=(
    StreamingWeightCacheProvider&& other) {
#define XNN_MOVE_MEMBER(member) member = std::move(other.member)
  XNN_MOVE_MEMBER(cache_provider_);
  // The contexts need to keep pointing to their owning object.
  cache_provider_.context = this;
  other.cache_provider_.context = &other;
  XNN_MOVE_MEMBER(file_path_);
  XNN_MOVE_MEMBER(buffer_address_to_identifier_);
  XNN_MOVE_MEMBER(buffer_remaps_);
  XNN_MOVE_MEMBER(cache_key_to_offset_);
  XNN_MOVE_MEMBER(mmap_handles_);
  XNN_MOVE_MEMBER(mmap_buffer_base_offset_);
  XNN_MOVE_MEMBER(file_descriptor_);
  XNN_MOVE_MEMBER(builder_);
  XNN_MOVE_MEMBER(building_run_);
  XNN_MOVE_MEMBER(is_build_step_);
  XNN_MOVE_MEMBER(offset_to_addr_);
#undef XNN_MOVE_MEMBER
  return *this;
}

void StreamingWeightCacheProvider::SetFilePath(const char* path) {
  XNNPACK_ABORT_CHECK(
      !IsBuilding(),
      "Cannot change the path of a cache that has already been loaded.");
  const char* const safe_path = Sanitize(path);
  if (file_path_ != safe_path) {
    // We try to keep file_path_'s data as stable as possible. Don't overwrite
    // if the path hasn't changed.
    file_path_ = safe_path;
  }
}

bool StreamingWeightCacheProvider::LoadOrStartBuild(const char* path,
                                               FileDescriptor fd) {
  if (!path && !fd.IsValid()) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "Cannot load or build XNNPack cache without specifying a "
                    "path or a file descriptor.");
    return false;
  }
  const char* const safe_path = Sanitize(path);
  FileDescriptor build_fd = fd.Duplicate();
  if (!IsInMemoryCachePath(safe_path) && Load(safe_path, std::move(fd))) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                    "XNNPack weight cache loaded from '%s'.", safe_path);
    return true;
  } else if (StartBuild(safe_path, std::move(build_fd))) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_VERBOSE,
                    "XNNPack weight cache build for '%s' started.", safe_path);
    return true;
  }
  return false;
}

bool StreamingWeightCacheProvider::StartBuild(const char* path, FileDescriptor fd) {
  const char* const safe_path = Sanitize(path);
  SetFilePath(safe_path);

  if (!fd.IsValid()) {
    if (IsInMemoryCachePath(file_path_)) {
      fd = CreateInMemoryFileDescriptor("XNNPack in-memory weight cache");
    } else {
      fd = FileDescriptor::Open(file_path_.c_str(), O_CREAT | O_TRUNC | O_RDWR,
                                0644);
    }
  }
  XNNPACK_RETURN_CHECK(fd.IsValid(), "could not open file ('%s'): %s.",
                       file_path_.c_str(), strerror(errno));
  file_descriptor_ = std::move(fd);
  building_run_ = builder_.Start(safe_path, file_descriptor_);
  return building_run_;
}

bool StreamingWeightCacheProvider::Load(const std::string& path, FileDescriptor fd) {
  SetFilePath(path.c_str());
  file_descriptor_ = std::move(fd);
  return Load();
}

bool StreamingWeightCacheProvider::Load() {
  mmap_buffer_base_offset_ = 0;
  cache_key_to_offset_.clear();
  mmap_handles_.resize(1);
  MMapHandle& mmap_handle = mmap_handles_.front();
  ScopeGuard unmap_on_fail([this] { mmap_handles_.clear(); });

  if (file_descriptor_.IsValid()) {
    XNNPACK_RETURN_CHECK(mmap_handle.Map(file_descriptor_,
                                         /*offset=*/0, file_path_.c_str()));
  } else {
    XNNPACK_ABORT_CHECK(!file_path_.empty(),
                        "Path wasn't provided to weight cache provider.");
    if (!FileExists(file_path_.c_str())) {
      TFLITE_LOG(tflite::TFLITE_LOG_WARNING,
                 "XNNPack weight cache: could not load '%s': %s.",
                 file_path_.c_str(), strerror(errno));
      return false;
    }
    XNNPACK_RETURN_CHECK(mmap_handle.Map(file_path_.c_str()));
  }

  XNNPACK_RETURN_CHECK(mmap_handle.size() >= sizeof(XNNPackCacheHeader),
                       "invalid cache file size: %zu, expected at least %zu.",
                       mmap_handle.size(), sizeof(XNNPackCacheHeader));

  const XNNPackCacheHeader header = [&mmap_handle] {
    XNNPackCacheHeader header;
    memcpy(&header, mmap_handle.data(), sizeof(header));
    return header;
  }();

  XNNPACK_RETURN_CHECK(header.version == XNNPackCacheHeader::kVersion,
                       "incompatible header version. Got %zd, expected %zd. "
                       "Cache needs to be built again.",
                       header.version, XNNPackCacheHeader::kVersion);

  XNNPACK_RETURN_CHECK(xnn_experimental_check_build_identifier(
                           header.xnnpack_build_identifier,
                           sizeof(header.xnnpack_build_identifier)),
                       "XNNPack weight cache: incompatible XNNPack version. "
                       "Cache needs to be built again.");

  XNNPACK_RETURN_CHECK(header.buffer_list_offset < mmap_handle.size(),
                       "invalid offset for buffer list descriptor.");

  XNNPACK_RETURN_CHECK(
      header.buffer_list_size == mmap_handle.size() - header.buffer_list_offset,
      "invalid size for buffer list descriptor.");

  // Verifiy the flabuffer part of the file.
  flatbuffers::Verifier verifier(mmap_handle.data() + header.buffer_list_offset,
                                 header.buffer_list_size);
  XNNPACK_RETURN_CHECK(cache::schema::VerifyBufferListBuffer(verifier),
                       "buffer list validation failed.");

  // Load flatbuffer.
  const cache::schema::BufferList* buffer_list = cache::schema::GetBufferList(
      mmap_handle.data() + header.buffer_list_offset);
  XNNPACK_RETURN_CHECK(buffer_list,
                       "could not get packed weights from flatbuffer.");

  mmap_buffer_base_offset_ = buffer_list->base_offset();
  if (const auto buffers = buffer_list->buffers(); buffers) {
    for (auto* buffer : *buffers) {
      XNNPACK_RETURN_CHECK(buffer, "invalid buffer address in buffer list.");
      cache_key_to_offset_.emplace(
          PackIdentifier{/*pack_algorithm_id=*/buffer->packing_algorithm_id(),
                         /*weights_id=*/buffer->weights_id(),
                         /*bias_id=*/buffer->bias_id()},
          BufferLocation{/*offset=*/buffer->offset(), /*size=*/buffer->size()});
      offset_to_addr_.insert(
          {buffer->offset(),
           mmap_handle.data() + mmap_buffer_base_offset_ + buffer->offset()});
    }
  }

  unmap_on_fail.Deactivate();
  return true;
}

bool StreamingWeightCacheProvider::LoadLastBuildStep() {
  if (mmap_handles_.empty()) {
    return Load();
  }

  if (builder_.LastBuildStepSize() == 0) {
    return true;
  }

  const XNNPackCacheHeader header = [this] {
    XNNPackCacheHeader header;
    memcpy(&header, mmap_handles_.front().data(), sizeof(header));
    return header;
  }();

  // Map last data segment:
  // - either resize the last mmap handle;
  // - or add a new mapping handle.
  {
    MMapHandle& last_mmap_handle = mmap_handles_.back();
    const int last_mmap_size = last_mmap_handle.size();
    if (!last_mmap_handle.Resize(last_mmap_size +
                                 builder_.LastBuildStepSize())) {
      mmap_handles_.emplace_back();
      if (file_descriptor_.IsValid()) {
        XNNPACK_RETURN_CHECK(
            mmap_handles_.back().Map(file_descriptor_,
                                     /*offset=*/builder_.LastBuildStepStart()),
            "could not map last build step");
      } else {
        XNNPACK_RETURN_CHECK(
            mmap_handles_.back().Map(file_path_.c_str(),
                                     /*offset=*/builder_.LastBuildStepStart()),
            "could not map last build step");
      }
    }
  }
  // Read the updated buffer list.
  MMapHandle& segment_mmap_handle = mmap_handles_.back();
  const size_t buffer_list_offset =
      header.buffer_list_offset - segment_mmap_handle.offset();

  flatbuffers::Verifier verifier(
      segment_mmap_handle.data() + buffer_list_offset, header.buffer_list_size);
  XNNPACK_RETURN_CHECK(cache::schema::VerifyBufferListBuffer(verifier),
                       "buffer list validation failed.");

  const cache::schema::BufferList* buffer_list = cache::schema::GetBufferList(
      segment_mmap_handle.data() + buffer_list_offset);
  XNNPACK_RETURN_CHECK(buffer_list,
                       "could not get packed weights from flatbuffer.");

  // Update offset_to_addr_ with new offsets
  const ptrdiff_t offset_modifier =
      buffer_list->base_offset() - segment_mmap_handle.offset();
  for (const auto* buffer : *(buffer_list->buffers())) {
    const size_t offset = buffer->offset();
    if (!offset_to_addr_.count(offset)) {
      offset_to_addr_.insert(
          {offset, segment_mmap_handle.data() + offset + offset_modifier});
    }
  }
  return true;
}

bool StreamingWeightCacheProvider::StartBuildStep() {
  XNNPACK_RETURN_CHECK(CanStartBuildStep(),
                       "cannot append data to an existing cache file.");
  if (IsBuilding()) {
    return true;
  }
  is_build_step_ = builder_.StartBuildStep();
  return is_build_step_;
}

bool StreamingWeightCacheProvider::StopBuildStep() {
  XNNPACK_RETURN_CHECK(builder_.StopBuildStep());
#if defined(_MSC_VER) || defined(XNNPACK_CACHE_NO_MMAP_FOR_TEST)
  if (!mmap_handles_.empty()) {
    // Sync mmap_handles_.data() with the content updated by
    // builder_.StopBuildStep().
    XNNPACK_RETURN_CHECK(file_descriptor_.IsValid());
    XNNPACK_RETURN_CHECK(mmap_handles_.front().Map(
        file_descriptor_, /*offset=*/0, file_path_.c_str()));
  }
#endif
  is_build_step_ = false;
  return LoadLastBuildStep();
}

void StreamingWeightCacheProvider::MapTensorIdentifiers(
    const TfLiteTensor* tensors, const size_t size,
    const std::unordered_map<size_t, size_t>& tensor_index_to_identifier) {
  for (const auto [index, identifier] : tensor_index_to_identifier) {
    XNNPACK_ABORT_CHECK(index < size,
                        "Tensor index corresponds to a non existing tensor.");
    buffer_address_to_identifier_[tensors[index].data.data] = identifier;
  }
}

void StreamingWeightCacheProvider::RemapDataBuffer(const void* const buffer,
                                              const void* const new_buffer) {
  buffer_remaps_[new_buffer] = buffer;
}

size_t StreamingWeightCacheProvider::LookUp(
    const xnn_weights_cache_look_up_key* cache_key) {
  if (!cache_key) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "XNNPack weight cache: a null cache key was provided.");
    return SIZE_MAX;
  }
  const PackIdentifier pack_id = BuildPackIdentifier(*cache_key);
  if (auto offset_it = cache_key_to_offset_.find(pack_id);
      offset_it != cache_key_to_offset_.end()) {
    return offset_it->second.offset;
  }
  return SIZE_MAX;
}

void* StreamingWeightCacheProvider::ReserveSpace(size_t size) {
  XNNPACK_ABORT_CHECK(IsBuilding(),
                      "Cannot reserve space in a cache that isn't building.");
  return builder_.Reserve(size);
}

size_t StreamingWeightCacheProvider::LookUpOrInsert(
    const xnn_weights_cache_look_up_key* cache_key, void* ptr, size_t size) {
  XNNPACK_ABORT_CHECK(cache_key, "A null cache key was provided.");

  const PackIdentifier pack_id = BuildPackIdentifier(*cache_key);
  if (auto offset_it = cache_key_to_offset_.find(pack_id);
      offset_it != cache_key_to_offset_.end()) {
    return offset_it->second.offset;
  }

  XNNPACK_ABORT_CHECK(
      IsBuilding(), "Cannot insert a buffer in a cache that is not building.");

  const BufferLocation location = builder_.Append(pack_id, ptr, size);
  XNNPACK_ABORT_CHECK(!location.IsInvalid(),
                      "Inserting data in the cache failed.");
  cache_key_to_offset_.emplace(pack_id, location);
  return location.offset;
}

void* StreamingWeightCacheProvider::OffsetToAddr(const size_t offset) {
  // While the cache is being built, the buffer could grow and need to be
  // reallocated so we cannot ensure pointer stability.
  XNNPACK_ABORT_CHECK(
      !IsBuilding(),
      "Cannot get the address of a buffer in a cache during a building step.");
  return offset_to_addr_[offset];
}

void StreamingWeightCacheProvider::Release() {
  buffer_address_to_identifier_.clear();
  cache_key_to_offset_.clear();
  mmap_handles_.clear();
  mmap_buffer_base_offset_ = 0;
  builder_ = WeightCacheBuilder();
}

size_t StreamingWeightCacheProvider::look_up(
    void* context, const xnn_weights_cache_look_up_key* cache_key) {
  return reinterpret_cast<StreamingWeightCacheProvider*>(context)->LookUp(cache_key);
}

void* StreamingWeightCacheProvider::reserve_space(void* context, size_t n) {
  return reinterpret_cast<StreamingWeightCacheProvider*>(context)->ReserveSpace(n);
}

size_t StreamingWeightCacheProvider::look_up_or_insert(
    void* context, const xnn_weights_cache_look_up_key* cache_key, void* ptr,
    size_t size) {
  return reinterpret_cast<StreamingWeightCacheProvider*>(context)->LookUpOrInsert(
      cache_key, ptr, size);
}

bool StreamingWeightCacheProvider::is_finalized(void* context) {
  return reinterpret_cast<StreamingWeightCacheProvider*>(context)->IsActive();
}

void* StreamingWeightCacheProvider::offset_to_addr(void* context, size_t offset) {
  return reinterpret_cast<StreamingWeightCacheProvider*>(context)->OffsetToAddr(
      offset);
}

enum xnn_status StreamingWeightCacheProvider::delete_cache(void* context) {
  reinterpret_cast<StreamingWeightCacheProvider*>(context)->Release();
  return xnn_status_success;
}

PackIdentifier StreamingWeightCacheProvider::BuildPackIdentifier(
    const xnn_weights_cache_look_up_key& key) {
  const auto get_buffer_id = [&](const void* buffer) -> size_t {
    if (buffer) {
      const auto identifier_it = buffer_address_to_identifier_.find(buffer);
      if (identifier_it != buffer_address_to_identifier_.end()) {
        return identifier_it->second;
      }
      // We could have several layers of remapping. We look through
      // buffer_remaps_ until we find a valid identifier or nothing is mapped to
      // the current buffer pointer.
      auto remapped_it = buffer_remaps_.find(buffer);
      while (remapped_it != buffer_remaps_.end()) {
        const auto remapped_identifier_it =
            buffer_address_to_identifier_.find(remapped_it->second);
        if (remapped_identifier_it != buffer_address_to_identifier_.end()) {
          return remapped_identifier_it->second;
        }
        remapped_it = buffer_remaps_.find(remapped_it->second);
      }
      XNNPACK_ABORT_CHECK(
          remapped_it != buffer_remaps_.end(),
          "Unknown constant buffer passed to BuildPackIdentifier.");
    }
    return PackIdentifier::kNoId;
  };
  return PackIdentifier{/*pack_algorithm_id=*/key.seed,
                        /*weights_id=*/get_buffer_id(key.kernel),
                        /*bias_id=*/get_buffer_id(key.bias)};
}



}  // namespace tflite::xnnpack
