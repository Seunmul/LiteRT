#include "tflite/delegates/xnnpack/weight_cache.h"

#include <fcntl.h>
#include <sys/stat.h>
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
#include <cstdlib>  // for std::getenv, std::strtoul
#include <unistd.h> // for direct io
#include <thread>   // for multithreading
#include <future>   // for std::future
#include <vector>   // for std::vector
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
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
    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                    "XNNPack weight cache loaded from '%s'.", safe_path);
    return true;
  } else if (StartBuild(safe_path, std::move(build_fd))) {
    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
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

bool StreamingWeightCacheProvider::Load(const std::string& path, 
                                        FileDescriptor fd) {
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
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                  "XNNPack weight cache mmap base offset: %zu",
                  mmap_buffer_base_offset_);
  
  // Load the buffer entries.
  if (const auto buffers = buffer_list->buffers(); buffers) {
    for (auto* buffer : *buffers) {
      XNNPACK_RETURN_CHECK(buffer, "invalid buffer address in buffer list.");
      cache_key_to_offset_.emplace(
          PackIdentifier{/*pack_algorithm_id=*/buffer->packing_algorithm_id(),
                         /*weights_id=*/buffer->weights_id(),
                         /*bias_id=*/buffer->bias_id()},
          BufferLocation{/*offset=*/buffer->offset(), /*size=*/buffer->size()});
      offset_to_addr_.insert({buffer->offset(), mmap_handle.data() + mmap_buffer_base_offset_ + buffer->offset()});
      offset_to_size_.insert({buffer->offset(), buffer->size()});
      offset_to_weights_id_.insert({buffer->offset(), buffer->weights_id()});
      
        //   TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
        //     "Loaded buffer: pack_algorithm_id=%zu, weights_id=%zu, bias_id=%zu, offset=%zu, size=%zu",
        //          buffer->packing_algorithm_id(),
        //          buffer->weights_id(),
        //          buffer->bias_id(),
        //          buffer->offset(),
        //          buffer->size());
    }
  }

  unmap_on_fail.Deactivate();

  return true;
}

void StreamingWeightCacheProvider::InitWeightChunkPrefetcher() {
  if (!weight_chunk_prefetcher_) {
    weight_chunk_prefetcher_ = std::make_unique<WeightChunkPrefetcher>();
    weight_chunk_prefetcher_->Init(this);
    
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                    "XNNPack weight cache prefetcher initialized");
  }
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

void StreamingWeightCacheProvider::MapTensorIdentifiers(const TfLiteTensor* tensors, 
    const size_t size, const std::unordered_map<size_t, size_t>& tensor_index_to_identifier) {

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

size_t StreamingWeightCacheProvider::LookUp(const xnn_weights_cache_look_up_key* cache_key) {

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

size_t StreamingWeightCacheProvider::LookUpByIds(size_t pack_algorithm_id,
                                                 size_t weights_id,
                                                 size_t bias_id) {
  PackIdentifier pid{ /*pack_algorithm_id=*/pack_algorithm_id,
                      /*weights_id=*/weights_id,
                      /*bias_id=*/bias_id };

  if (auto it = cache_key_to_offset_.find(pid); it != cache_key_to_offset_.end()) {
    return it->second.offset;
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

//! READY FOR IMPLEMENT DOUBLE BUFFERING -> We need to modify this function to return the address from the active buffer
        // if(weight_chunk_prefetcher_ == nullptr){
        //     TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR, 
        //         "OffsetToAddr: weight_chunk_prefetcher_ is not initialized!");
        //     return nullptr;
        // }

        // if (offset_to_weight_chunk_info_.find(offset) != offset_to_weight_chunk_info_.end()) {
        //     // 이미 로드된 청크인 경우, 해당 버퍼에서 주소 반환
        //     weight_chunk_info_t existing_chunk = offset_to_weight_chunk_info_.at(offset);
        //     return static_cast<uint8_t*>(managed_buffer_[existing_chunk.managed_buffer_index]) + existing_chunk.offset_adjust;
        // }

        // static size_t _chunk_index = 0;    
        // auto index_to_chunks = weight_chunk_prefetcher_->GetIndexToChunks();
        // weight_chunk_info_t weight_chunk_from_plan = index_to_chunks.at(_chunk_index);

        // if (weight_chunk_from_plan.origin_offset != offset){
        //     TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR, 
        //         "OffsetToAddr: chunk_index mismatch! expected offset=%zu, but got offset=%zu from plan",
        //         weight_chunk_from_plan.origin_offset, offset);
        //     return nullptr;
        // }

        // offset_to_weight_chunk_info_.insert({offset, weight_chunk_from_plan});                
        // _chunk_index++;

        
        // printf("OffsetToAddr:\n");

        // return static_cast<uint8_t*>(managed_buffer_[active_buffer_index_]) + weight_chunk_from_plan.offset_adjust;
        // }
//TODO: Fix non-aligned offset adjustment bug when calling OffsetToAddr in RUNTIME mode with prefetch plan created during PRE_RUNTIME mode
//Because of above bug, now we cannot use index_to_chunks loaded from PrefetchPlan file during OffsetToAddr in RUNTIME mode
void* StreamingWeightCacheProvider::OffsetToAddr(const size_t offset) {
    // While the cache is being built, the buffer could grow and need to be
    // reallocated so we cannot ensure pointer stability.
    XNNPACK_ABORT_CHECK(
        !IsBuilding(),
        "Cannot get the address of a buffer in a cache during a building step.");
     
    // weight streaming path   
    if (GetProviderMode() == ProviderMode::RUNTIME) {
        auto it = offset_to_size_.find(offset);
        void * ret_addr = nullptr;
        size_t buf_size = it->second;
        size_t abs_offset = mmap_buffer_base_offset_ + offset;
        
        static size_t _chunk_index = 0;
        

        if (offset_to_weight_chunk_info_.find(offset) != offset_to_weight_chunk_info_.end()) {
            // 이미 로드된 청크인 경우, 해당 버퍼에서 주소 반환
            weight_chunk_info_t existing_chunk = offset_to_weight_chunk_info_.at(offset);
            size_t offset_adjust = abs_offset - existing_chunk.aligned_offset;
            ret_addr = static_cast<uint8_t*>(managed_buffer_[existing_chunk.managed_buffer_index]) + offset_adjust;
        }else{
            size_t aligned_offset = (abs_offset / managed_buffer_sector_size_) * managed_buffer_sector_size_;
            size_t offset_adjust = abs_offset - aligned_offset;
            size_t aligned_size = ((buf_size + offset_adjust + managed_buffer_sector_size_ - 1) / managed_buffer_sector_size_) * managed_buffer_sector_size_;
        
            weight_chunk_info_t weight_chunk;   
            weight_chunk.chunk_index = _chunk_index;
            weight_chunk.aligned_size = aligned_size;
            weight_chunk.aligned_offset = aligned_offset;
            weight_chunk.origin_offset = offset;
            weight_chunk.origin_size = buf_size;
            weight_chunk.managed_buffer_index = active_buffer_index_;
            weight_chunk.weights_id = offset_to_weights_id_.at(offset);
            offset_to_weight_chunk_info_.insert({offset, weight_chunk});

            // printf("OffsetToAddr: chunk_index=%zu, aligned_offset=%zu, aligned_size=%zu, offset_adjust=%zu, buffer_index=%d, origin_offset=%zu, origin_size=%zu, weights_id=%zu\n",
            //     weight_chunk.chunk_index, weight_chunk.aligned_offset, weight_chunk.aligned_size, offset_adjust,
            //     weight_chunk.managed_buffer_index, weight_chunk.origin_offset, weight_chunk.origin_size, weight_chunk.weights_id);
                
            // Optional debug output
            // uint8_t* actual_data = static_cast<uint8_t*>(managed_buffer_[active_buffer_index_]) + offset_adjust;
            
            // if(memcmp(offset_to_addr_.at(offset), actual_data, buf_size) != 0) {
            //     printf("Data mismatch after pread for offset=%zu size=%zu\n", offset, buf_size);
            // } else {
            //     printf("Data match after pread for offset=%zu size=%zu\n", offset, buf_size);
            // }

            ret_addr = static_cast<uint8_t*>(managed_buffer_[active_buffer_index_]) + offset_adjust;
            _chunk_index++;
        }

        return ret_addr;

    }
    else if (GetProviderMode() == ProviderMode::PRE_RUNTIME)  // prefetch plan building path
    {

        auto it = offset_to_size_.find(offset);
        size_t buf_size = it->second;
        size_t abs_offset = mmap_buffer_base_offset_ + offset;
        
        // 이미 로드된 청크인 경우
        // weight_chunk_info_t existing_chunk = offset_to_weight_chunk_info_.at(offset);
        // ret_addr = static_cast<uint8_t*>(managed_buffer_[existing_chunk.managed_buffer_index]) + existing_chunk.offset_adjust;
    
        // 새로운 weight chunk인 경우에만, offset_to_weight_chunk_info_에 정보 추가
        if (offset_to_weight_chunk_info_.find(offset) == offset_to_weight_chunk_info_.end()) {
            size_t aligned_offset = (abs_offset / managed_buffer_sector_size_) * managed_buffer_sector_size_;
            size_t offset_adjust = abs_offset - aligned_offset;
            size_t aligned_size = ((buf_size + offset_adjust + managed_buffer_sector_size_ - 1) / managed_buffer_sector_size_) * managed_buffer_sector_size_;
        
            static size_t _chunk_index = 0;
            
            weight_chunk_info_t weight_chunk;
            weight_chunk.chunk_index = _chunk_index;
            weight_chunk.aligned_offset = aligned_offset;
            weight_chunk.offset_adjust = offset_adjust;
            weight_chunk.aligned_size = aligned_size;
            weight_chunk.origin_offset = offset;
            weight_chunk.origin_size = buf_size;
            weight_chunk.managed_buffer_index = active_buffer_index_;
            weight_chunk.weights_id = offset_to_weights_id_.at(offset);
            offset_to_weight_chunk_info_.insert({offset, weight_chunk});

            // printf("OffsetToAddr: chunk_index=%zu, aligned_offset=%zu, aligned_size=%zu, offset_adjust=%zu, buffer_index=%d, origin_offset=%zu, origin_size=%zu, weights_id=%zu\n",
            //     weight_chunk.chunk_index, weight_chunk.aligned_offset, weight_chunk.aligned_size, offset_adjust,
            //     weight_chunk.managed_buffer_index, weight_chunk.origin_offset, weight_chunk.origin_size, weight_chunk.weights_id);
            // active_buffer_index_ = 1 - active_buffer_index_; // switch buffer for next call  
            _chunk_index++;
        }
        return offset_to_addr_.at(offset);
    }
    else if(GetProviderMode() == ProviderMode::DEBUG_MMAP){ // general path(with mmap)
        return offset_to_addr_.at(offset);
    }
}


void StreamingWeightCacheProvider::PreInvokeHook(const size_t offset){

    if (GetProviderMode() == ProviderMode::RUNTIME) {      
        weight_chunk_prefetcher_->LoadWeightChunk(offset);
    }
    else if(GetProviderMode() == ProviderMode::PRE_RUNTIME) {
        // mmap을 사용하는 경우, prefetch plan 작성 중
        // weight_cache_provider_에서 필요한 정보 가져오기
        auto chunk_info_it = offset_to_weight_chunk_info_.find(offset);
        if (chunk_info_it == offset_to_weight_chunk_info_.end()) {
            return;
        }
        
        const auto& chunk_info = chunk_info_it->second;
 
        if (chunk_info_writer) {
            chunk_info_writer->WriteChunkInfo(chunk_info, weight_chunk_prefetcher_->GetPrefetcherMode());
        }
    }
    return; 
}

void StreamingWeightCacheProvider::PostInvokeHook(const size_t offset){
    
}

void StreamingWeightCacheProvider::Release() {
  buffer_address_to_identifier_.clear();
  cache_key_to_offset_.clear();
  mmap_handles_.clear();
  mmap_buffer_base_offset_ = 0;
  builder_ = WeightCacheBuilder();
}


// Open a file descriptor for direct I/O operations, with external file path
bool StreamingWeightCacheProvider::OpenDirectIOFileDescriptor(std::string file_path) {
    if (direct_io_file_descriptor_.IsValid()) {
        // Already opened
        return true;
    }

    direct_io_file_descriptor_ = FileDescriptor::Open(file_path.c_str(), O_RDONLY | O_DIRECT);
    XNNPACK_RETURN_CHECK(direct_io_file_descriptor_.IsValid(), "could not open file ('%s') for direct I/O: %s.",
                         file_path.c_str(), strerror(errno));
    return true;
}

bool StreamingWeightCacheProvider::CloseDirectIOFileDescriptor() {
    if (direct_io_file_descriptor_.IsValid()) {
        direct_io_file_descriptor_.Close();
    }
    return true;
}

void StreamingWeightCacheProvider::AllocManagedBuffer(size_t size) {
  managed_buffer_size_ = size;

  // 보통 4096 정렬 (파일시스템 블록 사이즈 기준)
  size_t aligned_size = ((size + managed_buffer_sector_size_ - 1) / managed_buffer_sector_size_) * managed_buffer_sector_size_;

  for (int i = 0; i < 2; ++i) {
    if (posix_memalign(&managed_buffer_[i], managed_buffer_sector_size_, aligned_size) != 0) {
      perror("posix_memalign failed");
      managed_buffer_[i] = nullptr;
      managed_buffer_size_ = 0;
      return;
    }
    // Zero-fill
    memset(managed_buffer_[i], 0, size);
  }
}

void StreamingWeightCacheProvider::FreeManagedBuffer() {
  for (int i = 0; i < 2; ++i) {
    if (managed_buffer_[i]) {
      free(managed_buffer_[i]);
      managed_buffer_[i] = nullptr;
    }
  }
  managed_buffer_size_ = 0;
}

void StreamingWeightCacheProvider::SwitchActiveBuffer() {
  active_buffer_index_ = 1 - active_buffer_index_;
}

void StreamingWeightCacheProvider::ResetActiveBuffer() {
  active_buffer_index_ = 0;
}

/********* Utilities *********/

void* StreamingWeightCacheProvider::GetMmappedAddr(size_t offset) {
  auto it = offset_to_addr_.find(offset);
  if (it != offset_to_addr_.end()) return it->second;
  return nullptr;
}

bool StreamingWeightCacheProvider::DumpWeightCacheStructureToFile(const std::string& dump_file_path) const {
  std::ofstream file(dump_file_path);
  if (!file.is_open()) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "Failed to open dump file: %s", dump_file_path.c_str());
    return false;
  }
  
  file << DumpWeightCacheStructure();
  file.close();
  
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                  "Cache structure dumped to: %s", dump_file_path.c_str());
  return true;
}

std::string StreamingWeightCacheProvider::DumpWeightCacheStructure() const {
  std::ostringstream oss;
  
  // 헤더 정보 출력
  oss << "=== XNNPack Weight Cache Structure Dump ===\n\n";
  
  if (mmap_handles_.empty()) {
    oss << "Cache not loaded.\n";
    return oss.str();
  }
  
  const MMapHandle& mmap_handle = mmap_handles_.front();
  
  // 파일 기본 정보
  oss << "File Information:\n";
  oss << "  Total file size: " << mmap_handle.size() << " bytes (" 
      << std::fixed << std::setprecision(2) << (mmap_handle.size() / 1024.0 / 1024.0) << " MB)\n";
  oss << "  File path: " << file_path_ << "\n\n";
  
  // 헤더 정보 추출 및 출력
  if (mmap_handle.size() < sizeof(XNNPackCacheHeader)) {
    oss << "File too small to contain valid header.\n";
    return oss.str();
  }
  
  XNNPackCacheHeader header;
  memcpy(&header, mmap_handle.data(), sizeof(header));
  
  oss << "Header Information:\n";
  oss << "  Version: " << header.version << " (expected: " << XNNPackCacheHeader::kVersion << ")\n";
  oss << "  XNNPack build identifier: ";
  for (size_t i = 0; i < sizeof(header.xnnpack_build_identifier); ++i) {
    oss << std::hex << std::setw(2) << std::setfill('0') 
        << static_cast<unsigned>(header.xnnpack_build_identifier[i]);
  }
  oss << std::dec << "\n";
  oss << "  Buffer list offset: " << header.buffer_list_offset << " bytes (" 
      << std::fixed << std::setprecision(2) << (header.buffer_list_offset / 1024.0 / 1024.0) << " MB)\n";
  oss << "  Buffer list size: " << header.buffer_list_size << " bytes\n";
  oss << "  Data section size: " << (header.buffer_list_offset - sizeof(header)) << " bytes ("
      << std::fixed << std::setprecision(2) << ((header.buffer_list_offset - sizeof(header)) / 1024.0 / 1024.0) << " MB)\n\n";
  
  // FlatBuffer 파싱 (검증은 스킵)
  if (header.buffer_list_offset >= mmap_handle.size() ||
      header.buffer_list_size != mmap_handle.size() - header.buffer_list_offset) {
    oss << "Invalid buffer list offset/size.\n";
    return oss.str();
  }
  
  const cache::schema::BufferList* buffer_list = cache::schema::GetBufferList(
      mmap_handle.data() + header.buffer_list_offset);
  
  if (!buffer_list) {
    oss << "Failed to parse BufferList from FlatBuffer.\n";
    return oss.str();
  }
  
  // BufferList 정보 출력
  oss << "FlatBuffer BufferList Information:\n";
  oss << "  Base offset: " << buffer_list->base_offset() << " bytes\n";
  
  const auto buffers = buffer_list->buffers();
  if (!buffers) {
    oss << "  No buffers found.\n";
    return oss.str();
  }
  
  oss << "  Number of buffers: " << buffers->size() << "\n\n";
  
  // 각 버퍼 정보 출력
  oss << "Buffer Details:\n";
  // 설명 추가: base_offset, mmap base pointer, 메모리 범위 계산 방식
  oss << "  NOTE: This dump reflects the weight cache FILE layout (static) and the process mapping / runtime addresses (dynamic).\n";
  oss << "        Columns marked [STATIC] come from the on-disk FlatBuffer file;\n";
  oss << "        Columns marked [DYNAMIC] are process/runtime values (vary by execution).\n";
  oss << "  Note: 'Offset' is buffer->offset(), relative to BufferList.base_offset() [STATIC].\n";
  oss << "        'Memory Range' is [base_offset + offset, base_offset + offset + size - 1] (file byte offsets) [STATIC].\n";
  oss << "        'Mmapped Addr' is the process virtual address where the file was mapped: mmap_base_ptr + (base_offset + offset) [DYNAMIC].\n";
  // mmap base pointer (헥스) 출력
  {
    std::ostringstream mbp;
    mbp << "0x" << std::hex << std::setfill('0') << std::setw(12)
        << reinterpret_cast<uintptr_t>(mmap_handle.data());
    oss << "  mmap base pointer: " << mbp.str() << " (process virtual address base)\n";
  }
  // Indicate whether cache is file-backed or in-memory special path
  oss << "  Cache type: " << (IsInMemoryCachePath(file_path_) ? "IN-MEMORY (special)" : "FILE-BACKED") << "\n";
  oss << std::left << std::setfill(' ')  // fill을 공백으로 명시적 설정
      << std::setw(6) << "Index"
      << " | " << std::setw(12) << "Weight ID"  
      << " | " << std::setw(12) << "Pack Algo ID"
      << " | " << std::setw(12) << "Bias ID"
      << " | " << std::setw(10) << "Offset"
      << " | " << std::setw(10) << "Size"
      << " | " << std::setw(22) << "Mmapped Addr [DYNAMIC]"
      << " | " << std::setw(16) << "Memory Range [DYNAMIC]\n";
  oss << std::string(120, '-') << "\n";
  
  for (size_t i = 0; i < buffers->size(); ++i) {
    const auto* buffer = buffers->Get(i);
    if (!buffer) {
      oss << std::setw(6) << i << " | NULL BUFFER\n";
      continue;
    }
    
    const uint64_t abs_offset = buffer_list->base_offset() + buffer->offset();
    const void* addr = mmap_handle.data() + abs_offset;
    
    // Bias ID 표시 처리
    std::string bias_str = (buffer->bias_id() == PackIdentifier::kNoId) ? "None" : std::to_string(buffer->bias_id());
    
    // 헥스 주소를 별도로 포맷팅
    std::ostringstream addr_stream;
    addr_stream << "0x" << std::hex << std::setfill('0') << std::setw(12) << reinterpret_cast<uintptr_t>(addr);
    
    oss << std::left << std::setfill(' ')  // fill을 공백으로 명시적 설정
        << std::setw(6) << i
        << " | " << std::setw(12) << buffer->weights_id()
        << " | " << std::setw(12) << buffer->packing_algorithm_id()
        << " | " << std::setw(12) << bias_str
        << " | " << std::setw(10) << buffer->offset()
        << " | " << std::setw(10) << buffer->size()
        << " | " << std::setw(22) << addr_stream.str()
        << " | [" << abs_offset << "-" << (abs_offset + buffer->size() - 1) << "]\n";
  }
  
  oss << "\n";
  
  
  oss << "Internal Cache State in Weight Cache:\n";
  oss << "  cache_key_to_offset_ entries: " << cache_key_to_offset_.size() << "\n";
  oss << "  offset_to_addr_ entries: " << offset_to_addr_.size() << "\n";
  oss << "  mmap_buffer_base_offset_: " << mmap_buffer_base_offset_ << "\n";
  oss << "  Number of mmap handles: " << mmap_handles_.size() << "\n\n";
  
  if (!cache_key_to_offset_.empty()) {
    oss << "Cache Key Mappingss :\n";
    oss << std::left << std::setfill(' ')  
    << std::setw(12) << "Weight ID"
    << " | " << std::setw(12) << "Pack Algo ID"
    << " | " << std::setw(12) << "Bias ID"
    << " | " << std::setw(10) << "Offset"
    << " | " << std::setw(10) << "Size\n";
    oss << std::string(75, '-') << "\n";
    
    size_t count = 0;
    for (const auto& [pack_id, location] : cache_key_to_offset_) {
      std::string bias_str = (pack_id.bias_id == PackIdentifier::kNoId) ? "None" : std::to_string(pack_id.bias_id);
      
      oss << std::left << std::setfill(' ')  
          << std::setw(12) << pack_id.weights_id
          << " | " << std::setw(12) << pack_id.pack_algorithm_id
          << " | " << std::setw(12) << bias_str
          << " | " << std::setw(10) << location.offset
          << " | " << std::setw(10) << location.size << "\n";
      ++count;
    }
  }
  
  return oss.str();
}

bool StreamingWeightCacheProvider::DumpTensorIdentifierMapToFile(const std::string& dump_file_path) const {
  std::ofstream file(dump_file_path);
  if (!file.is_open()) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                    "Failed to open dump file: %s", dump_file_path.c_str());
    return false;
  }

  file << DumpTensorIdentifierMap();
  file.close();
  
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                  "Cache structure dumped to: %s", dump_file_path.c_str());
  return true;
}

std::string StreamingWeightCacheProvider::DumpTensorIdentifierMap() const {
  std::ostringstream oss;
  oss << "=== Tensor Address → WeightCache Identifier Map ===\n";
  if (buffer_address_to_identifier_.empty()) {
    oss << " (empty)\n";
    return oss.str();
  }

  for (const auto& [addr, identifier] : buffer_address_to_identifier_) {
    oss << "  Addr: " << addr 
        << " -> Identifier: " << identifier << "\n";
  }
  return oss.str();
}

bool StreamingWeightCacheProvider::VerifyBuffer(size_t offset) {
    auto size_it = offset_to_size_.find(offset);
    if (size_it == offset_to_size_.end()) {
        fprintf(stdout, "[VerifyBuffer] No size found for offset=%zu\n", offset);
        return false;
    }
    size_t buf_size = size_it->second;

    // mmap 기반 원본 데이터
    const uint8_t* mmap_ptr = reinterpret_cast<const uint8_t*>(
        offset_to_addr_.at(offset));

    // pread 기반 streaming 데이터 with O_DIRECT alignment
    size_t abs_offset = mmap_buffer_base_offset_ + offset;
    
    // O_DIRECT requires sector alignment (usually 512 bytes)
    size_t aligned_offset = (abs_offset / managed_buffer_sector_size_) * managed_buffer_sector_size_;
    size_t offset_adjust = abs_offset - aligned_offset;
    size_t aligned_size = ((buf_size + offset_adjust + managed_buffer_sector_size_ - 1) / managed_buffer_sector_size_) * managed_buffer_sector_size_;
    
    // Allocate aligned buffer
    void* aligned_buffer = nullptr;
    if (posix_memalign(&aligned_buffer, managed_buffer_sector_size_, aligned_size) != 0) {
        perror("[VerifyBuffer] posix_memalign failed");
        return false;
    }
    
    int fd = open(file_path_.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("[VerifyBuffer] open failed");
        free(aligned_buffer);
        return false;
    }
    
    ssize_t n = pread(fd, aligned_buffer, aligned_size, aligned_offset);
    close(fd);
    
    if (n < 0 || static_cast<size_t>(n) != aligned_size) {
        perror("[VerifyBuffer] pread failed");
        free(aligned_buffer);
        return false;
    }
    
    // Extract the actual data from aligned buffer
    uint8_t* actual_data = static_cast<uint8_t*>(aligned_buffer) + offset_adjust;

    // 비교
    if (memcmp(mmap_ptr, actual_data, buf_size) == 0) {
        printf("[VerifyBuffer] offset=%zu size=%zu : MATCH ✅\n", offset, buf_size);
        //show first/last few bytes
        printf("First few bytes(mmap_ptr):\n");
        for (size_t i = 0; i < std::min<size_t>(20, buf_size); i++) {
            printf("0x%02x ", mmap_ptr[i]);
        }
        printf("\n");
        printf("First few bytes(pread):\n");
        for (size_t i = 0; i < std::min<size_t>(20, buf_size); i++) {
            printf("0x%02x ", actual_data[i]);
        }
        printf("\n");
        printf("Last few bytes(mmap_ptr):\n");
        for (size_t i = buf_size-1; i > buf_size - 20; i--) {
            printf("0x%02x ", mmap_ptr[i]);
        }
        printf("\n");
        printf("Last few bytes(pread):\n");
        for (size_t i = buf_size-1; i > buf_size - 20; i--) {
            printf("0x%02x ", actual_data[i]);
        }
        printf("\n");

        free(aligned_buffer);
        return true;
    } else {
        printf("[VerifyBuffer] offset=%zu size=%zu : MISMATCH ❌\n", offset, buf_size);
        // optional: dump first few mismatched bytes
        for (size_t i = 0; i < std::min<size_t>(64, buf_size); i++) {
            if (mmap_ptr[i] != actual_data[i]) {
                printf("  mismatch at byte %zu: mmap=0x%02x pread=0x%02x\n",
                       i, mmap_ptr[i], actual_data[i]);
                break;
            }
        }
        free(aligned_buffer);
        return false;
    }
}

bool StreamingWeightCacheProvider::VerifyAllBuffers() {
    bool all_match = true;
    for (const auto& [offset, _] : offset_to_addr_) {
        if (!VerifyBuffer(offset)) {
            all_match = false;
        }
    }
    return all_match;
}


/**************
 * Privates
 **************/

/************ C Interfaces ************/

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

void StreamingWeightCacheProvider::pre_invoke_hook(void* context, size_t offset) {
  reinterpret_cast<StreamingWeightCacheProvider*>(context)->PreInvokeHook(offset);
}

void StreamingWeightCacheProvider::post_invoke_hook(void* context, size_t offset) {
  reinterpret_cast<StreamingWeightCacheProvider*>(context)->PostInvokeHook(offset);
}

/************ Internal Methods ************/

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

//* ============ WeightChunkPrefetcher ============ */


void WeightChunkPrefetcher::Init(StreamingWeightCacheProvider* provider) {
  weight_cache_provider_ = provider;
  prefetch_mode_ = PrefetchMode::UNINITIALIZED; // 기본값
}

void WeightChunkPrefetcher::SetPrefetchPlan(
  PrefetchMode mode,
  std::unordered_map<size_t, size_t>&& offset_to_index,
  std::vector<StreamingWeightCacheProvider::weight_chunk_info_t>&& chunks) {
  const int idx = ModeToIndex(mode);
  if (idx < 0) return;
  plans_[idx].offset_to_index = std::move(offset_to_index);
  plans_[idx].chunks = std::move(chunks);  
  has_plan_[idx] = true;
}

bool WeightChunkPrefetcher::HasPlan(PrefetchMode mode) const {
  const int idx = ModeToIndex(mode);
  return idx >= 0 && has_plan_[idx];
}

const WeightChunkPrefetcher::PrefetchPlan* WeightChunkPrefetcher::GetPlan(PrefetchMode mode) const {
  const int idx = ModeToIndex(mode);
  if (idx < 0 || !has_plan_[idx]) return nullptr;
  return &plans_[idx];
}

void WeightChunkPrefetcher::BuildIndexToChunksFromPlans() {
  // Compute maximum chunk_index across all plans so we can size the vector.
  size_t max_index = 0;
  bool seen_any = false;
  for (int i = 0; i < 2; ++i) {
    if (!has_plan_[i]) continue;
    for (const auto& ch : plans_[i].chunks) {
      if (!seen_any || ch.chunk_index > max_index) max_index = ch.chunk_index;
      seen_any = true;
    }
  }

  if (!seen_any) {
    index_to_chunks_.clear();
    return;
  }

  // Initialize with sentinel entries (chunk_index == SIZE_MAX marks empty).
  StreamingWeightCacheProvider::weight_chunk_info_t sentinel;
  sentinel.chunk_index = SIZE_MAX;
  sentinel.aligned_offset = 0;
  sentinel.offset_adjust = 0;
  sentinel.aligned_size = 0;
  sentinel.origin_offset = 0;
  sentinel.origin_size = 0;
  sentinel.managed_buffer_index = -1;
  sentinel.weights_id = 0;

  index_to_chunks_.assign(max_index + 1, sentinel);

  // Populate, keeping first-seen entry on conflicts.
  for (int i = 0; i < 2; ++i) {
    if (!has_plan_[i]) continue;
    for (const auto& ch : plans_[i].chunks) {
      const size_t idx = ch.chunk_index;
      auto& dest = index_to_chunks_[idx];
      if (dest.chunk_index == SIZE_MAX) {
        dest = ch;
      } else if (dest.origin_offset != ch.origin_offset) {
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                        "WeightChunkPrefetcher::BuildIndexToChunksFromPlans: conflict for chunk_index=%zu: existing origin_offset=%zu, new origin_offset=%zu. Keeping existing.",
                        idx, dest.origin_offset, ch.origin_offset);
      }
    }
  }
}


bool WeightChunkPrefetcher::LoadWeightChunk(size_t offset) {
    
  if (prefetch_mode_ == PrefetchMode::UNINITIALIZED) {
    return false;
  }
  if (!HasPlan(prefetch_mode_)) {
    return false;
  }  
  const auto& plan = plans_[ModeToIndex(prefetch_mode_)];
  auto it = plan.offset_to_index.find(offset);
  if (it == plan.offset_to_index.end()) return false;
  size_t idx = it->second;  

  if (idx >= index_to_chunks_.size()) return false;
  const auto& chunk_info = index_to_chunks_.at(idx);
  
  uint8_t* target_ptr = static_cast<uint8_t*>(
    weight_cache_provider_->managed_buffer_[weight_cache_provider_->active_buffer_index_]);
  
  ssize_t bytes_read = pread(
    weight_cache_provider_->direct_io_file_descriptor_.Value(),
    target_ptr,
    chunk_info.aligned_size,
    chunk_info.aligned_offset
  );
  
  return bytes_read > 0 && static_cast<size_t>(bytes_read) == chunk_info.aligned_size;
}


}  // namespace tflite::xnnpack
