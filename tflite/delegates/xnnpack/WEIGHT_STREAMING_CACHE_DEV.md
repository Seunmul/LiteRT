# Streaming Weight Cache Development Log

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [설계 철학](#설계-철학)
- [현재 구현 상황](#현재-구현-상황)
- [통합 Double Buffering 아키텍처](#통합-double-buffering-아키텍처)
- [기술적 도전과제](#기술적-도전과제)
- [성능 목표](#성능-목표)
- [파일 구조](#파일-구조)
- [다음 단계 계획](#다음-단계-계획)
- [XNNPACK 연동 워크플로우](#xnnpack-weight-cache-연동-워크플로우-분석)
- [참고 자료](#참고-자료)

## 프로젝트 개요

TFLite XNNPACK Delegate의 weight cache 시스템을 기존 mmap 기반에서 on-demand streaming 방식으로 확장하는 프로젝트.

**핵심 목표:**

- 메모리 사용량 최소화 (필요한 weight만 로드)
- 기존 빌드/저장 로직 100% 호환성 유지
- 점진적 도입 가능한 composition 설계

## 설계 철학

### Composition over Inheritance

**선택한 방향:** Composition 패턴  
**이유:**

- 기존 `MMapWeightCacheProvider` 코드 Zero 수정
- 의존성 최소화 (빌드 시에만 MMap provider 생성)
- 명확한 책임 분리 (빌드 vs 스트리밍)
- ABI 호환성 완벽 유지

### 모드별 전략

```
┌─────────────────┬─────────────────┬─────────────────┐
│     모드        │    빌드 작업    │    로딩 작업    │
├─────────────────┼─────────────────┼─────────────────┤
│ Build Mode      │ MMap Delegate   │ MMap Delegate   │
│ Streaming Mode  │ N/A             │ Custom Stream   │
└─────────────────┴─────────────────┴─────────────────┘
```

## 구현 단계

### ✅ Phase 1: Composition 설계 (완료)

**작업 내용:**

- `StreamingWeightCacheProvider` 클래스 설계
- Composition 패턴으로 `MMapWeightCacheProvider` 내포
- 모드별 라우팅 메커니즘 설계

**핵심 구조:**

```cpp
class StreamingWeightCacheProvider {
private:
    // 빌드 모드: 필요시에만 생성
    std::unique_ptr<MMapWeightCacheProvider> build_provider_;

    // 스트리밍 모드: 경량화된 상태
    FileDescriptor streaming_fd_;
    std::unordered_map<size_t, BufferLocation> stream_index_;
    std::unordered_map<size_t, MaterializedBlob> materialized_blobs_;
};
```

### ✅ Phase 2: 빌드 메서드 위임 (완료)

**구현된 기능:**

1. **지연 생성 (Lazy Initialization)**

   ```cpp
   void EnsureBuildProvider() {
       if (!build_provider_) {
           build_provider_ = std::make_unique<MMapWeightCacheProvider>();
       }
   }
   ```

2. **빌드 메서드 위임**

   - `StartBuild()` → `build_provider_->StartBuild()`
   - `StartBuildStep()` → `build_provider_->StartBuildStep()`
   - `StopBuildStep()` → `build_provider_->StopBuildStep()`
   - `LookUpOrInsert()` → `build_provider_->LookUpOrInsert()`

3. **하이브리드 라우팅**

   ```cpp
   void* OffsetToAddr(size_t offset) {
       if (IsInBuildMode()) {
           return build_provider_->OffsetToAddr(offset);  // MMap
       }
       return StreamingOffsetToAddr(offset);              // Streaming
   }
   ```

4. **XNNPACK 인터페이스 준수**
   - 모든 static callback 함수들 구현
   - `xnn_weights_cache_provider` 구조체 제공

## 🎯 현재 구현 상황 및 향후 계획

### ✅ 완료된 구현

1. **Composition 기반 아키텍처**

   - `StreamingWeightCacheProvider`가 `MMapWeightCacheProvider` 포함
   - 빌드/저장 로직은 MMap 방식 그대로 유지

2. **빌드 메서드 위임 완료**

   - 모든 build 관련 메서드들이 내부 `MMapWeightCacheProvider`로 완벽 위임
   - 기존 cache 생성/저장 로직 100% 호환

3. **XNNPACK 인터페이스 호환성**
   - 모든 static callback 함수 구현
   - 기존 코드 수정 없이 drop-in replacement 가능

### 🚧 진행 중인 작업

#### **통합 Double Buffering 아키텍처 (최종 설계)**

**핵심 설계 결정:**

- **하나의 통합된 메커니즘**: 가상 주소와 큰 고정 버퍼를 분리된 전략이 아닌 하나로 구현
- **하드코딩 MAX_TENSOR_SIZE**: 모델의 최대 텐서 크기를 사전에 설정하고 2개 고정 버퍼 할당
- **XNNPACK Offset Hooking**: 가상 주소 공간 예약으로 cache offset을 실제 버퍼 주소로 매핑
- **MMap Fallback 제거**: 우선 핵심 구현에 집중, fallback은 나중에 추가

**통합 아키텍처 설계:**

```cpp
class StreamingDoubleBuffer {
private:
    // 하드코딩된 설정 (실제 환경에 맞게 조정 가능)
    static constexpr size_t MAX_TENSOR_SIZE = 512 * 1024 * 1024;  // 512MB
    static constexpr size_t ALIGNMENT = 4096;  // 4KB for O_DIRECT

    struct Buffer {
        void* data;               // 4KB 정렬된 고정 메모리 (MAX_TENSOR_SIZE)
        size_t capacity;          // Buffer capacity (= MAX_TENSOR_SIZE)
        size_t valid_size;        // 현재 로딩된 데이터 크기
        size_t file_offset;       // 현재 버퍼가 담고 있는 파일 위치
        bool is_active;           // XNNPACK이 현재 사용 중
        bool is_loading;          // 백그라운드 로딩 진행 중
    };

    // 핵심: 2개 고정 버퍼로 Double Buffering
    Buffer buffers_[2];
    int active_buffer_;           // XNNPACK이 READ하는 버퍼 (0 or 1)
    int loading_buffer_;          // SSD에서 WRITE하는 버퍼 (1-active_buffer)

    // 가상 주소 공간 hooking으로 포인터 안정성 보장
    void* virtual_base_;          // 전체 cache 크기만큼 가상 주소 예약
    size_t total_cache_size_;     // 전체 cache 파일 크기

    // File I/O
    int cache_fd_;                // O_DIRECT로 열린 파일 디스크립터

public:
    // XNNPACK 호환성: cache offset -> 안정적인 포인터 반환
    void* GetStablePointer(size_t cache_offset) {
        // 1. 가상 주소로 항상 동일한 포인터 반환 (포인터 안정성)
        void* virtual_ptr = static_cast<uint8_t*>(virtual_base_) + cache_offset;

        // 2. 해당 offset이 현재 활성 버퍼에 있는지 확인
        if (IsOffsetInActiveBuffer(cache_offset)) {
            return virtual_ptr;  // 이미 로딩됨
        }

        // 3. 필요하면 버퍼 스왑 후 로딩
        SwapBuffersAndLoad(cache_offset);
        return virtual_ptr;
    }

    // Double Buffering 핵심 로직
    bool SwapBuffersAndLoad(size_t target_offset) {
        // 1. 로딩 버퍼 완료 대기
        WaitForLoadingComplete();

        // 2. 안전하게 버퍼 스왑
        active_buffer_ = loading_buffer_;
        loading_buffer_ = 1 - active_buffer_;

        // 3. 새로운 활성 버퍼를 가상 주소에 매핑
        MapBufferToVirtualAddress(target_offset);

        // 4. 다음 청크 백그라운드 로딩 시작
        StartBackgroundLoading(GetNextOffset(target_offset));

        return true;
    }

private:
    bool Initialize(const std::string& cache_file_path) {
        // 1. 전체 cache 크기만큼 가상 주소 공간 예약
        total_cache_size_ = GetCacheFileSize(cache_file_path);
        virtual_base_ = mmap(nullptr, total_cache_size_, PROT_NONE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        // 2. 2개 고정 버퍼 할당 (4KB 정렬, O_DIRECT 호환)
        for (int i = 0; i < 2; i++) {
            buffers_[i].data = aligned_alloc(ALIGNMENT, MAX_TENSOR_SIZE);
            buffers_[i].capacity = MAX_TENSOR_SIZE;
            buffers_[i].is_active = false;
            buffers_[i].is_loading = false;
        }

        // 3. O_DIRECT로 파일 열기
        cache_fd_ = open(cache_file_path.c_str(), O_RDONLY | O_DIRECT);

        return virtual_base_ != MAP_FAILED && cache_fd_ >= 0;
    }

    void MapBufferToVirtualAddress(size_t file_offset) {
        // 현재 활성 버퍼의 데이터를 해당 가상 주소에 매핑
        void* virtual_addr = static_cast<uint8_t*>(virtual_base_) + file_offset;
        Buffer& active = buffers_[active_buffer_];

        // 기존 매핑 해제
        munmap(virtual_addr, active.valid_size);

        // 새로운 데이터를 가상 주소에 매핑
        mmap(virtual_addr, active.valid_size, PROT_READ,
             MAP_PRIVATE | MAP_FIXED, /* temp_fd */, 0);
        // 실제로는 버퍼 데이터를 해당 가상 주소로 복사하거나
        // 더 효율적인 방법으로 매핑
    }
};
```

**StreamingWeightCacheProvider 통합:**

```cpp
class StreamingWeightCacheProvider : public WeightCacheProvider {
private:
    std::unique_ptr<MMapWeightCacheProvider> build_provider_;
    std::unique_ptr<StreamingDoubleBuffer> double_buffer_;

public:
    void* OffsetToAddr(size_t offset) override {
        if (IsInBuildMode()) {
            return build_provider_->OffsetToAddr(offset);  // 빌드 모드
        }

        // 스트리밍 모드: 통합 Double Buffering
        return double_buffer_->GetStablePointer(offset);
    }

    bool PrepareStreaming(const std::string& cache_file_path) {
        double_buffer_ = std::make_unique<StreamingDoubleBuffer>();
        return double_buffer_->Initialize(cache_file_path);
    }
};
```

**핵심 장점:**

1. **단순성**: 복잡한 전략 분기 없이 하나의 통합된 구현
2. **포인터 안정성**: 가상 주소 hooking으로 XNNPACK 호환성 완벽 보장
3. **메모리 효율성**: 2 × MAX_TENSOR_SIZE로 고정, 예측 가능한 메모리 사용량
4. **성능**: Double Buffering으로 동시 READ/WRITE, 백그라운드 스트리밍
5. **구현 간소화**: MMap fallback 제거로 핵심 로직에 집중

### 🎯 다음 단계 TODO

1.  **통합 StreamingDoubleBuffer 구현**

    - 하드코딩된 MAX_TENSOR_SIZE 기반 2개 고정 버퍼 할당
    - 가상 주소 공간 예약 및 cache offset hooking 구현
    - Double buffering 상태 관리 (active/loading buffer 스왑)

2.  **XNNPACK Offset Hooking 메커니즘**

    - `GetStablePointer(cache_offset)` 구현
    - 가상 주소 → 실제 버퍼 데이터 매핑 로직
    - 포인터 안정성 보장 검증

3.  **백그라운드 로딩 시스템**

    - O_DIRECT pread를 이용한 비동기 데이터 로딩
    - Buffer 스왑 시점 최적화
    - 에러 처리 및 복구 메커니즘

4.  **성능 측정 및 최적화**

    - 메모리 사용량 측정 (2 × MAX_TENSOR_SIZE 고정)
    - Cold start 시간 측정 (기존 mmap 대비)
    - I/O 패턴 분석 및 최적화
      for (int i = 0; i < 2; i++) {
      buffers[i].data = aligned_alloc(4096, buffer_capacity);
      buffers[i].capacity = buffer_capacity;
      buffers[i].state = AVAILABLE;
      }

          // O_DIRECT로 파일 열기
          fd = open(cache_file.c_str(), O_RDONLY | O_DIRECT);
          return fd >= 0;

      }

    ```

    ```

5.  **비동기 로딩 시스템**

    ```cpp
    bool LoadTensorToBuffer(size_t offset, size_t size, int buffer_idx) {
        Buffer& buf = buffers[buffer_idx];
        if (buf.state != AVAILABLE) return false;

        // 섹터 정렬된 읽기
        size_t aligned_offset = AlignDown(offset, 512);
        size_t aligned_size = AlignUp(offset + size - aligned_offset, 512);

        buf.state = LOADING;
        ssize_t bytes = pread(fd, buf.data, aligned_size, aligned_offset);

        if (bytes > 0) {
            buf.size = size;
            buf.file_offset = offset;
            buf.state = READY;
            return true;
        }

        buf.state = AVAILABLE;
        return false;
    }
    ```

6.  **Pointer Stability 보장**

    ```cpp
    void* StreamingOffsetToAddr(size_t offset) {
        // 1. 현재 활성 버퍼에서 찾기
        Buffer& active = buffers[active_buffer];
        if (active.state == READY &&
            offset >= active.file_offset &&
            offset < active.file_offset + active.size) {
            return static_cast<uint8_t*>(active.data) +
                   (offset - active.file_offset);
        }

        // 2. 필요하면 로딩 수행
        int next_buffer = 1 - active_buffer;
        if (LoadTensorToBuffer(offset, GetTensorSize(offset), next_buffer)) {
            active_buffer = next_buffer;
            return StreamingOffsetToAddr(offset);  // 재귀 호출
        }

        return nullptr;  // 로딩 실패
    }
    ```

**Double Buffering 워크플로우:**

1. **초기화**: 2개 4KB-정렬 버퍼 할당, O_DIRECT 파일 열기
2. **첫 번째 레이어**: Buffer A에 로딩, XNNPACK에 포인터 반환
3. **레이어 실행 중**: Buffer B에 다음 레이어 백그라운드 로딩
4. **레이어 전환**: Buffer 스왑, A는 AVAILABLE로 변경
5. **반복**: 계속해서 앞서 로딩하며 스왑

## 기술적 도전과제

### 해결된 문제들

1. **FileDescriptor Move Semantics**

   - 문제: `FileDescriptor`가 copy 불가
   - 해결: `std::move()` 사용

2. **Include Path 이슈**

   - 문제: FlatBuffers 헤더 경로 문제
   - 임시 해결: 해당 부분 주석 처리, 나중에 수정 예정

3. **스트리밍 방식 선택**
   - 문제: 단순 heap allocation vs mmap vs 익명 메모리 예약
   - 해결: **익명 메모리 + pread 방식 채택** (포인터 안정성 + 메모리 효율성)

### 새로운 Double Buffering 방식의 핵심 과제들

1. **O_DIRECT 정렬 요구사항**

   - 도전: 메모리 4KB 정렬, 파일 오프셋 512B 정렬, 읽기 크기 섹터 단위
   - 해결책: `aligned_alloc(4096, buffer_size)` + `AlignDown/AlignUp` 함수
   - 주의사항: XNNPACK 128B 정렬과 4KB 정렬 조화 필요

2. **버퍼 상태 관리**

   - 도전: LOADING/READY/AVAILABLE 상태 간 안전한 전환
   - 해결 방안:
     - **원자적 상태 전환**: 상태 변경 전 데이터 준비 완료 보장
     - **백그라운드 로딩**: 현재 레이어 실행 중 다음 레이어 미리 로딩
     - **버퍼 스왑 시점**: XNNPACK 레이어 경계에서만 수행

3. **포인터 안정성 보장 (핵심 도전)** ⚠️

   **XNNPACK의 포인터 사용 패턴:**

   ```cpp
   // 1. Setup Phase: XNNPACK이 weights 포인터 요청
   void* weight_ptr = cache_provider->offset_to_addr(context, offset);

   // 2. XNNPACK 내부 구조체에 포인터 저장
   struct xnn_node {
       void* weights_data;  // Setup에서 받은 포인터 저장
   };

   // 3. Runtime Phase: 저장된 포인터를 그대로 사용 (재호출 없음!)
   float* weights = static_cast<float*>(node->weights_data);
   // ⚠️ 이 포인터가 무효화되면 crash!
   ```

   **Double Buffering의 핵심 문제:**

   - 버퍼 스왑 시 이전에 반환한 포인터들이 모두 무효화
   - XNNPACK은 포인터 무효화를 모르고 계속 사용
   - 결과: 메모리 접근 오류, 데이터 corruption

4. **포인터 안정성 보장 (핵심 도전)** ⚠️

   **XNNPACK의 포인터 사용 패턴:**

   ```cpp
   // 1. Setup Phase: XNNPACK이 weights 포인터 요청
   void* weight_ptr = cache_provider->offset_to_addr(context, offset);

   // 2. XNNPACK 내부 구조체에 포인터 저장
   struct xnn_node {
       void* weights_data;  // Setup에서 받은 포인터 저장
   };

   // 3. Runtime Phase: 저장된 포인터를 그대로 사용 (재호출 없음!)
   float* weights = static_cast<float*>(node->weights_data);
   // ⚠️ 이 포인터가 무효화되면 crash!
   ```

   **통합 Double Buffering의 해결책:**

   **🎯 핵심: 가상 주소 Hooking + 고정 버퍼 Double Buffering**

   ```cpp
   // 1. 가상 주소 공간 예약으로 포인터 안정성 보장
   void* virtual_base = mmap(nullptr, total_cache_size, PROT_NONE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

   void* GetStablePointer(size_t cache_offset) {
       // 항상 동일한 가상 주소 반환 (포인터 안정성 완벽 보장)
       return static_cast<uint8_t*>(virtual_base) + cache_offset;
   }

   // 2. 하드코딩된 2개 고정 버퍼로 Double Buffering
   struct Buffer {
       void* data;               // aligned_alloc(4096, 512MB) 고정 할당
       size_t capacity;          // = MAX_TENSOR_SIZE (512MB)
       size_t file_offset;       // 현재 버퍼가 담고 있는 파일 위치
       bool is_active;           // XNNPACK 사용 중
   };
   Buffer buffers[2];            // A, B 버퍼

   // 3. 버퍼 스왑 시 가상 주소 매핑만 변경 (포인터는 그대로)
   bool SwapBuffersAndLoad(size_t target_offset) {
       // Buffer A에서 Buffer B로 스왑
       active_buffer = loading_buffer;
       loading_buffer = 1 - active_buffer;

       // 새로운 활성 버퍼를 동일한 가상 주소에 매핑
       MapBufferToVirtualAddress(target_offset);

       // XNNPACK이 저장한 포인터는 여전히 유효!
       // 가상 주소는 그대로, 내용만 새로운 버퍼 데이터로 변경됨
   }
   ```

   **핵심 장점:**

   - **포인터 안정성**: XNNPACK이 받은 포인터가 절대 변경되지 않음
   - **메모리 효율성**: 2 × 512MB = 1GB 고정 (기존 대비 80-90% 절약)
   - **단순한 구현**: 복잡한 전략 분기 없이 하나의 통합된 메커니즘
   - **예측 가능성**: 하드코딩된 크기로 메모리 사용량 완전 예측 가능

   ```cpp
   // 2개 고정 버퍼로 동시 읽기/사용 처리
   struct DoubleBuffer {
       struct Buffer {
           void* data;           // 4KB 정렬된 메모리
           size_t capacity;      // 버퍼 크기
           size_t valid_size;    // 현재 유효한 데이터 크기
           size_t file_offset;   // 파일에서의 시작 위치
           bool is_active;       // XNNPACK이 현재 사용 중인지
           bool is_loading;      // 백그라운드에서 로딩 중인지
       };

       Buffer buffers[2];        // A, B 버퍼
       int active_buffer;        // 현재 XNNPACK이 사용하는 버퍼 (0 or 1)
       int loading_buffer;       // 현재 SSD에서 로딩하는 버퍼 (1-active_buffer)
   };

   void* GetWeightPointer(size_t offset) {
       Buffer& active = buffers[active_buffer];

       // 현재 활성 버퍼에서 offset이 유효한 범위에 있으면 반환
       if (offset >= active.file_offset &&
           offset < active.file_offset + active.valid_size) {
           return static_cast<uint8_t*>(active.data) + (offset - active.file_offset);
       }

       // 범위를 벗어나면 버퍼 스왑 필요
       SwapBuffersWhenSafe(offset);
       return GetWeightPointer(offset);  // 재귀 호출
   }

   void SwapBuffersWhenSafe(size_t target_offset) {
       // 1. 로딩 버퍼가 target_offset을 포함하고 있는지 확인
       Buffer& loading = buffers[loading_buffer];
       if (loading.is_loading) {
           WaitForLoadingComplete();  // 로딩 완료 대기
       }

       if (target_offset >= loading.file_offset &&
           target_offset < loading.file_offset + loading.valid_size) {
           // 2. 안전하게 버퍼 스왑
           active_buffer = loading_buffer;
           loading_buffer = 1 - active_buffer;

           // 3. 새로운 로딩 버퍼에서 다음 청크 미리 로딩 시작
           StartBackgroundLoading(GetNextChunkOffset());
       }
   }
   ```

   **핵심 동작 원리:**

   - **동시 처리**: XNNPACK이 Buffer A 사용 중에 Buffer B에서 다음 데이터 로딩
   - **포인터 안정성**: 활성 버퍼는 XNNPACK 사용 완료까지 절대 변경 안됨
   - **백그라운드 스트리밍**: SSD 읽기와 추론 연산이 병렬로 진행
   - **Zero-copy 스왑**: 버퍼 인덱스만 변경, 데이터 복사 없음

   **🥈 방안 2: 가상 주소 공간 예약 (메모리 효율적)**

   ```cpp
   // 전체 cache 크기만큼 가상 주소 예약, 물리 메모리는 필요한 부분만 매핑
   void* virtual_base = mmap(nullptr, total_cache_size, PROT_NONE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

   void* GetStablePointer(size_t offset) {
       return static_cast<uint8_t*>(virtual_base) + offset;  // 항상 동일한 주소
   }

   bool MapPhysicalWindow(size_t offset, size_t size) {
       void* target_addr = static_cast<uint8_t*>(virtual_base) + offset;
       // 해당 가상 주소에 실제 데이터 매핑 (포인터 불변, 내용만 변경)
   }
   ```

   **� 방안 3: 큰 고정 버퍼 (실용적)**

   ```cpp
   // 가장 큰 레이어를 담을 수 있는 충분한 버퍼
   size_t max_layer_size = CalculateMaxLayerSize();  // 예: 200MB
   size_t buffer_capacity = max_layer_size * 2;     // 400MB 고정 할당

   // 레이어 전체가 단일 버퍼에 들어가므로 포인터 안정성 보장
   // 레이어 간 전환 시에만 버퍼 스왑
   ```

   **🥉 방안 3: MMap Fallback (안전망)**

   ```cpp
   // 위 방법들이 실패하면 기존 방식 사용
   if (virtual_address_failed || buffer_too_large || platform_unsupported) {
       return LoadWithMMapFallback();  // 100% 안전한 기존 방식
   }
   ```

   **Hybrid 구현 전략:**

   ```cpp
   enum PointerStabilityStrategy {
       VIRTUAL_ADDRESS_SPACE,  // 64-bit, 큰 모델용
       LARGE_FIXED_BUFFER,     // 작은-중간 모델용
       MMAP_FALLBACK          // 실패 시 안전망
   };

   PointerStabilityStrategy DetermineStrategy() {
       if (is_64_bit && total_cache_size < 16GB) {
           return VIRTUAL_ADDRESS_SPACE;
       }
       if (max_layer_size * 2 < 500MB) {
           return LARGE_FIXED_BUFFER;
       }
       return MMAP_FALLBACK;
   }
   ```

5. **성능 최적화**

   - 메모리 사용량: 고정 2×MAX_TENSOR_SIZE (예측 가능)
   - I/O 지연시간: O_DIRECT로 OS 캐시 우회, 하지만 정렬 오버헤드
   - CPU 사용량: 백그라운드 스레드로 미리 로딩, 메인 스레드 블록 최소화

6. **에러 처리 및 Fallback**
   - O_DIRECT 지원하지 않는 파일시스템: 일반 pread로 fallback
   - 메모리 할당 실패: 버퍼 크기 줄여서 재시도
   - I/O 에러: 기존 MMapWeightCacheProvider로 fallback

### 성능 특성 분석

**메모리 사용량 예상:**

```
기존 mmap 방식: 전체 cache 파일 크기 (예: 2-4GB 모델)
새 방식 가상 메모리: 동일 (수 GB)
새 방식 물리 메모리: 10-20% 수준 (200-400MB)
```

**지연 시간 분석:**

- Cold start: 헤더만 읽으므로 90%+ 단축
- Runtime pread: 순차 읽기 시 mmap page fault와 거의 동등
- 메모리 해제: `madvise(DONTNEED)` 즉시 반영 vs 수요 페이징 지연

**시스템 리소스:**

- 파일 디스크립터: 1개 (기존과 동일)
- 시스템콜 횟수: 증가 (pread 다수 vs mmap 1회), 하지만 page fault 대신
- CPU 사용량: 약간 증가 (직접 I/O), 하지만 예측 가능
  - I/O 실패 시 fallback 메커니즘
  - 부분 읽기 처리
  - 캐시 파일 corruption 대응

## 성능 목표

### Double Buffering 방식 성능 목표

**메모리 효율성:**

- **고정 메모리**: 2 × MAX_TENSOR_SIZE (완전 예측 가능)
- **전형적 사용량**: 200-400MB (기존 대비 80-90% 절약)
- **메모리 피크**: 버퍼 스왑 시에도 고정 (예측 가능한 최악 시나리오)

**성능 특성:**

- **Cold start**: 인덱스 + 첫 버퍼만 로딩, 95%+ 초기화 시간 단축
- **Runtime latency**: O_DIRECT로 OS 캐시 우회, 일관된 I/O 성능
- **백그라운드 로딩**: 현재 레이어 실행 중 다음 레이어 미리 준비

**확장성:**

- **대형 모델**: 텐서 크기와 무관하게 고정 메모리 사용
- **멀티 모델**: 각 모델당 고정된 작은 메모리 풋프린트
- **임베디드 환경**: 제한된 메모리에서도 대형 모델 실행 가능

### 측정 계획

1. **메모리 사용량 추적**

   ```bash
   # RSS 변화 모니터링
   while true; do
       ps -o pid,rss,vsz,comm -p $PID
       sleep 1
   done

   # 가상 vs 물리 메모리 상세 분석
   cat /proc/$PID/status | grep -E "(VmSize|VmRSS|VmData)"
   ```

2. **초기화 시간 벤치마킹**

   ```cpp
   // Phase별 시간 측정
   auto t1 = timer(); LoadIndexOnly(); auto t2 = timer();
   auto t3 = timer(); ReserveAllVirtualRegions(); auto t4 = timer();

   printf("Index: %dus, Reserve: %dus\n",
          duration_us(t1, t2), duration_us(t3, t4));
   ```

3. **런타임 지연 시간**

   ```cpp
   // 레이어별 로딩 시간
   auto start = timer();
   EnsureLoaded(offset, size);
   auto load_time = duration_us(start, timer());

   // 실제 연산 시간과 분리 측정
   ```

4. **시스템 리소스 모니터링**
   - `strace -e pread`: 실제 pread 호출 패턴 확인
   - `perf stat`: page fault vs system call 비율
   - `/proc/meminfo`: 전역 메모리 압박 영향 분석

## 파일 구조

```
tflite/delegates/xnnpack/
├── weight_cache.h                          # 기존 MMapWeightCacheProvider
├── weight_cache.cc                         # 기존 구현
├── streaming_weight_cache_unified.h        # 통합 아키텍처 헤더 ✅
├── streaming_weight_cache_unified.cc       # 통합 아키텍처 구현 (TODO)
├── streaming_weight_cache.h                # 기존 Streaming 헤더 (composition 버전)
├── streaming_weight_cache.cc               # 기존 Streaming 구현 (composition 버전)
└── WEIGHT_STREAMING_CACHE_DEV.md           # 이 개발 문서 ✅
```

## 성능 목표

### 통합 Double Buffering 방식 성능 목표

**메모리 효율성:**

- **고정 메모리**: 2 × MAX_TENSOR_SIZE = 1GB (완전 예측 가능)
- **전형적 사용량**: 1GB 고정 (기존 2-4GB 대비 70-85% 절약)
- **메모리 피크**: 항상 1GB 고정 (예측 가능한 최악 시나리오)

**성능 특성:**

- **Cold start**: 헤더 + 첫 버퍼만 로딩, 95%+ 초기화 시간 단축
- **Runtime latency**: O_DIRECT로 OS 캐시 우회, 일관된 I/O 성능
- **백그라운드 로딩**: 현재 텐서 사용 중 다음 텐서 미리 준비

**확장성:**

- **대형 모델**: 텐서 크기와 무관하게 1GB 고정 메모리
- **멀티 모델**: 각 모델당 1GB 고정된 작은 메모리 풋프린트
- **임베디드 환경**: 제한된 메모리에서도 대형 모델 실행 가능

## 다음 단계 계획

### 🎯 다음 단계 TODO

1. **통합 StreamingDoubleBuffer 구현**

   - 하드코딩된 MAX_TENSOR_SIZE 기반 2개 고정 버퍼 할당
   - 가상 주소 공간 예약 및 cache offset hooking 구현
   - Double buffering 상태 관리 (active/loading buffer 스왑)

2. **XNNPACK Offset Hooking 메커니즘**

   - `GetStablePointer(cache_offset)` 구현
   - 가상 주소 → 실제 버퍼 데이터 매핑 로직
   - 포인터 안정성 보장 검증

3. **백그라운드 로딩 시스템**

   - O_DIRECT pread를 이용한 비동기 데이터 로딩
   - Buffer 스왑 시점 최적화
   - 에러 처리 및 복구 메커니즘

4. **성능 측정 및 최적화**

   - 메모리 사용량 측정 (2 × MAX_TENSOR_SIZE 고정)
   - Cold start 시간 측정 (기존 mmap 대비)
   - I/O 패턴 분석 및 최적화

5. **Virtual Address Space Manager 구현** (우선순위 1)

   ```cpp
   class VirtualAddressManager {
       void* virtual_base_;
       size_t virtual_size_;
       std::map<size_t, bool> mapped_regions_;
   public:
       bool ReserveVirtualSpace(size_t total_cache_size);
       void* GetStablePointer(size_t offset);
       bool MapPhysicalWindow(size_t offset, size_t size);
       bool UnmapPhysicalWindow(size_t offset, size_t size);
   };
   ```

   - PROT_NONE으로 가상 주소 공간 예약
   - offset → 고정 가상 주소 변환
   - 동적 물리 메모리 매핑/해제
   - 메모리 사용량 추적 및 최적화

6. **Large Fixed Buffer Manager 구현** (우선순위 2)

   ```cpp
   class LargeBufferManager {
       void* buffer_a_;
       void* buffer_b_;
       size_t buffer_capacity_;
       int active_buffer_;
   public:
       bool InitBuffers(size_t max_layer_size);
       void* GetLayerPointer(size_t layer_offset, size_t layer_size);
       bool LoadNextLayer(size_t offset, size_t size);
   };
   ```

   - `CalculateMaxLayerSize()`: 캐시 인덱스 분석으로 최대 레이어 크기 계산
   - O_DIRECT 정렬된 대용량 버퍼 할당
   - 레이어 단위 로딩으로 포인터 안정성 보장

7. **MMap Fallback 시스템**
   - 위 방법들 실패 시 기존 `MMapWeightCacheProvider`로 우아한 전환
   - 에러 핸들링 및 성능 모니터링

### 단기 (3-5일) - 통합 및 최적화

1. **StreamingWeightCacheProvider 통합**

   ```cpp
   void* StreamingWeightCacheProvider::OffsetToAddr(size_t offset) {
       switch (pointer_stability_strategy_) {
           case VIRTUAL_ADDRESS_SPACE:
               return virtual_manager_.GetStablePointer(offset);
           case LARGE_FIXED_BUFFER:
               return buffer_manager_.GetLayerPointer(offset, GetTensorSize(offset));
           case MMAP_FALLBACK:
               return build_provider_->OffsetToAddr(offset);
       }
   }
   ```

2. **성능 측정 및 검증**

   - **포인터 안정성 테스트**: XNNPACK 실행 중 포인터 유효성 검증
   - **메모리 사용량 비교**: MMap vs Virtual Address vs Large Buffer
   - **Cold start 성능**: 각 전략별 초기화 시간 측정

3. **동적 전략 전환**
   - 런타임 중 메모리 압박 상황에서 전략 조정
   - 성능 모니터링 기반 최적 전략 학습

### 중기 (1-2주) - 고급 최적화

1. **메모리 압박 상황 대응**

   - Virtual Address Space의 물리 메모리 동적 해제
   - Large Buffer의 적응적 크기 조절
   - OOM 상황에서 MMap Fallback 자동 전환

2. **백그라운드 최적화**
   - 비사용 가상 메모리 영역 미리 해제
   - 다음 레이어 예측 및 prefetch
   - I/O 패턴 학습 및 최적화

## 코드 리뷰 체크리스트

### 포인터 안정성 중심 검토 항목

**🎯 핵심 검증 사항:**

- 🔍 **XNNPACK 포인터 안정성**: Setup phase에서 받은 포인터가 Runtime까지 유효한지 확인
- 🔍 **가상 주소 예약 성공률**: 다양한 환경에서 PROT_NONE mmap 성공 여부
- 🔍 **물리 메모리 매핑 효율성**: 동적 매핑/해제가 성능에 미치는 영향
- 🔍 **Large Buffer 크기 최적화**: 메모리 사용량 vs 포인터 안정성 트레이드오프
- 🔍 **Fallback 전환 안정성**: 전략 실패 시 MMap으로 우아한 전환 확인
- 🔍 **멀티스레드 안전성**: 동시 접근 시 포인터 무효화 방지

**✅ 검증 완료 항목:**

- ✅ Composition 패턴을 통한 기존 코드 Zero 수정
- ✅ 빌드/저장 워크플로우 100% 호환성 유지
- ✅ FlatBuffer 파일 포맷 완전 호환
- ✅ XNNPACK callback 인터페이스 정확한 구현

- 🔍 **O_DIRECT 플랫폼 지원**: Linux/Android에서 O_DIRECT 동작 확인
- 🔍 **정렬 요구사항 준수**: 4KB 메모리, 512B 오프셋, 섹터 크기 읽기
- 🔍 **버퍼 크기 최적화**: MAX_TENSOR_SIZE 자동 감지 및 설정
- 🔍 **포인터 안정성**: XNNPACK 실행 중 버퍼 스왑 금지 확인
- 🔍 **백그라운드 로딩**: 별도 스레드에서 안전한 미리 로딩
- 🔍 **fallback 메커니즘**: O_DIRECT 실패 시 일반 pread로 우아한 전환

## XNNPACK Weight Cache 연동 워크플로우 분석

### 전체 실행 흐름

XNNPACK weight cache가 실제 node 추론 시에 어떻게 연동되는지 단계별로 상세 분석:

```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│    Phase        │   TFLite        │   XNNPACK       │   Weight Cache  │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ 1. Delegate     │ Create Delegate │ -               │ Load/StartBuild │
│    Initialization│                │                 │                 │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ 2. Subgraph     │ Plan execution  │ Define operators│ LookUpOrInsert  │
│    Preparation  │                 │ & weights       │ (Build Mode)    │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ 3. Runtime      │ Create runtime  │ Create runtime  │ Finalize cache  │
│    Creation     │                 │ with cache      │                 │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ 4. Setup Phase  │ Setup tensors   │ Resolve weights │ OffsetToAddr    │
│                 │                 │ pointers        │ (Setup Mode)    │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ 5. Inference    │ Invoke          │ Execute nodes   │ Direct pointer  │
│    Runtime      │                 │                 │ access          │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

### Phase 1: Delegate Initialization

**TFLite Delegate 생성:**

```cpp
// 사용자 코드
TfLiteXNNPackDelegateOptions options = TfLiteXNNPackDelegateOptionsDefault();
options.weights_cache_file_path = "/path/to/cache.tflite_cache";
auto delegate = TfLiteXNNPackDelegateCreate(&options);

// 내부 구현
Delegate::Delegate(const TfLiteXNNPackDelegateOptions& options) {
    // Weight cache provider 초기화
    if (options.weights_cache != nullptr) {
        weight_cache_provider_ = static_cast<MMapWeightCacheProvider*>(options.weights_cache);
    } else if (!options.weights_cache_file_path.empty()) {
        // 파일 경로로부터 weight cache 로드/생성
        weight_cache_provider_->LoadOrStartBuild(options.weights_cache_file_path);
    }
}
```

**StreamingWeightCacheProvider 초기화:**

```cpp
bool StreamingWeightCacheProvider::LoadOrStartBuild(const char* file_path, FileDescriptor fd) {
    // 1. 기존 캐시 파일 존재 확인
    if (FileExists(file_path)) {
        return Load(file_path, std::move(fd));  // 스트리밍 모드로 로드
    } else {
        return StartBuild(file_path, std::move(fd));  // 빌드 모드로 시작
    }
}

bool StreamingWeightCacheProvider::Load(const std::string& path, FileDescriptor fd) {
    // 포인터 안정성 전략 결정
    pointer_stability_strategy_ = DetermineOptimalStrategy();

    switch (pointer_stability_strategy_) {
        case VIRTUAL_ADDRESS_SPACE:
            return LoadWithVirtualAddressSpace(path, std::move(fd));
        case LARGE_FIXED_BUFFER:
            return LoadWithLargeBuffer(path, std::move(fd));
        case MMAP_FALLBACK:
            return LoadWithMMapFallback(path, std::move(fd));
    }
}
```

### Phase 2: Subgraph Preparation

**XNNPACK Subgraph 생성 및 Operator 정의:**

```cpp
// TFLite Delegate Plan 단계
TfLiteStatus Subgraph::Plan(TfLiteContext* context, const TfLiteDelegateParams* params) {
    // 1. XNNPACK subgraph 생성
    xnn_subgraph_t subgraph = nullptr;
    xnn_create_subgraph(/*num_external_values=*/inputs.size() + outputs.size(),
                       /*flags=*/0, &subgraph);

    // 2. Weight cache가 활성화된 경우 build step 시작
    if (delegate.weight_cache_provider_->IsActive() &&
        delegate.weight_cache_provider_->CanStartBuildStep()) {
        delegate.weight_cache_provider_->StartBuildStep();
    }

    // 3. 각 operator를 XNNPACK subgraph에 정의
    for (auto node : planned_nodes) {
        DefineXNNPackNode(subgraph, node, delegate);
    }

    // 4. Build step 완료
    if (delegate.weight_cache_provider_->CanStartBuildStep()) {
        delegate.weight_cache_provider_->StopBuildStep();
    }
}
```

**Weight Cache 빌드 과정 (첫 번째 실행):**

```cpp
// 예: Convolution operator 정의 시
void DefineConvolutionNode(xnn_subgraph_t subgraph, int node_id, const Delegate& delegate) {
    // 1. Weight 데이터 준비
    const float* weights_data = GetTensorData(weights_tensor);
    size_t weights_size = GetTensorSize(weights_tensor);

    // 2. Weight cache에서 중복 확인 및 저장
    xnn_weights_cache_look_up_key cache_key = CreateCacheKey(weights_data, weights_size);

    // 3. StreamingWeightCacheProvider를 통한 처리
    size_t offset = delegate.weight_cache_provider_->LookUpOrInsert(&cache_key,
                                                                   (void*)weights_data,
                                                                   weights_size);

    // 4. XNNPACK operator 정의 (offset 저장)
    xnn_define_convolution_2d(...,
                             /*weights_id=*/offset,  // 실제 포인터가 아닌 offset 저장
                             ...);
}
```

### Phase 3: Runtime Creation

**XNNPACK Runtime 생성:**

```cpp
TfLiteStatus Subgraph::Invoke(TfLiteContext* context) {
    // 1. Weight cache를 포함한 runtime 생성
    xnn_runtime_t runtime = nullptr;
    xnn_status status = xnn_create_runtime_v4(subgraph.get(),
                                              delegate.weights_cache(),  // 우리의 cache provider
                                              delegate.workspace(),
                                              delegate.threadpool(),
                                              flags,
                                              &runtime);

    // 2. Runtime 생성 중 XNNPACK이 weight cache 초기화
    // XNNPACK 내부에서 cache provider의 is_finalized() 호출하여 상태 확인
}
```

### Phase 4: Setup Phase (포인터 해결)

**XNNPACK Setup 단계에서 포인터 요청:**

```cpp
// XNNPACK 내부 - Runtime Setup
xnn_status xnn_setup_runtime(xnn_runtime_t runtime, ...) {
    for (auto& node : runtime->nodes) {
        if (node.type == xnn_node_type_convolution) {
            // 1. Weight cache에서 실제 포인터 요청
            void* weights_ptr = runtime->weights_cache->offset_to_addr(
                runtime->weights_cache->context,
                node.weights_offset  // Phase 2에서 저장된 offset
            );

            // 2. 포인터를 node 구조체에 저장 (⚠️ 여기서 포인터 안정성 중요!)
            node.weights_data = weights_ptr;

            // 3. XNNPACK kernel setup
            setup_convolution_kernel(&node, weights_ptr, ...);
        }
    }
}
```

**StreamingWeightCacheProvider의 포인터 제공:**

```cpp
void* StreamingWeightCacheProvider::offset_to_addr(void* context, size_t offset) {
    StreamingWeightCacheProvider* provider = static_cast<StreamingWeightCacheProvider*>(context);

    // 포인터 안정성 전략에 따른 처리
    switch (provider->pointer_stability_strategy_) {
        case VIRTUAL_ADDRESS_SPACE:
            // 가상 주소 공간에서 안정적인 포인터 반환
            return provider->virtual_manager_.GetStablePointer(offset);

        case LARGE_FIXED_BUFFER:
            // 큰 고정 버퍼에서 레이어별 포인터 반환
            return provider->buffer_manager_.GetLayerPointer(offset);

        case MMAP_FALLBACK:
            // 기존 mmap 방식으로 fallback
            return provider->build_provider_->OffsetToAddr(offset);
    }
}
```

### Phase 5: Inference Runtime

**실제 추론 실행:**

```cpp
// TFLite Invoke
xnn_status status = xnn_invoke_runtime(runtime.get());

// XNNPACK 내부 실행
xnn_status xnn_invoke_runtime(xnn_runtime_t runtime) {
    for (auto& node : runtime->nodes) {
        // Setup phase에서 저장된 포인터 직접 사용
        float* weights = static_cast<float*>(node.weights_data);

        // ⚠️ 이 포인터가 여전히 유효해야 함!
        // StreamingWeightCacheProvider가 포인터 안정성을 보장해야 하는 이유

        execute_convolution_kernel(input, weights, output, ...);
    }
}
```

### 핵심 포인터 안정성 시나리오

**시나리오 1: 진정한 Double Buffering 방식**

```cpp
// Setup Phase - 초기 버퍼 A에 첫 번째 청크 로딩
buffers[0].LoadChunk(offset_0, chunk_size);
buffers[0].is_active = true;
active_buffer = 0;

void* weight_ptr = GetWeightPointer(offset_0);
node.weights_data = weight_ptr;  // XNNPACK이 Buffer A 포인터 저장

// Runtime Phase - XNNPACK이 Buffer A 사용 중에 Buffer B에서 다음 청크 로딩
StartBackgroundLoading(buffers[1], next_offset);  // 병렬로 SSD 읽기

// 다음 레이어 전환 시 - 안전한 버퍼 스왑
WaitForLoadingComplete(buffers[1]);
active_buffer = 1;  // 포인터는 이미 Buffer A를 가리키고 있어서 안전
```

**핵심 장점:**

- XNNPACK이 Buffer A 사용 중에 Buffer B로 다음 데이터 스트리밍
- 추론과 I/O가 완전 병렬 처리
- 포인터 안정성: 활성 버퍼는 절대 변경 안됨

**시나리오 2: 큰 고정 버퍼 방식**

```cpp
// Setup Phase - 전체 레이어를 단일 버퍼에 로딩
buffer_manager_.LoadCompleteLayer(layer_id);
void* layer_ptr = buffer_manager_.GetLayerPointer(offset);
node.weights_data = layer_ptr;

// Runtime Phase - 동일 레이어 내에서는 포인터 안정성 보장
float* weights = static_cast<float*>(node.weights_data);  // 안전
```

### 메모리 효율성 달성

**기존 MMap 방식:**

- Setup Phase: 전체 파일 mmap (2-4GB)
- Runtime Phase: 동일한 메모리 사용량

**새로운 Streaming 방식:**

- Setup Phase: 필요한 레이어만 로딩 (200-400MB)
- Runtime Phase: 동일한 적은 메모리 사용량
- 80-90% 메모리 절약 달성

이 워크플로우를 통해 XNNPACK의 포인터 안정성 요구사항을 만족시키면서도 메모리 효율성을 크게 개선할 수 있습니다.

## 참고 자료

- [WEIGHT_CACHE_README.md](./WEIGHT_CACHE_README.md) - 기존 시스템 분석
- [weight_cache.h](./weight_cache.h) - MMapWeightCacheProvider 인터페이스
- [weight_cache.cc](./weight_cache.cc) - 기존 구현 참고
- [DOUBLE_BUFFER_COMPATIBILITY.md](./DOUBLE_BUFFER_COMPATIBILITY.md) - 호환성 분석
- [POINTER_STABILITY_ANALYSIS.md](./POINTER_STABILITY_ANALYSIS.md) - 포인터 안정성 상세 분석
