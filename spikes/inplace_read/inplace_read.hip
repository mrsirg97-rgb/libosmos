// spike (design step): can the 8060S read a GGUF's expert bytes where they lie - the page cache, through mmap and
// hipHostRegister - as fast as from memory it allocated itself? the answer decides whether the memory planner needs a
// staging ring (maya's flow, ~1.4 GB/s here) or reads non-resident experts in place.
//
//   inplace_read <file> [offset_gb=8] [size_gb=2] [reps=5]
//   inplace_read <file> <offset_gb> <size_gb> cold
//
// prints GB/s for one read of `size` bytes by a streaming kernel: A device memory (hipMalloc), B pinned host memory
// (hipHostMalloc), C the file's pages registered in place (warm: in the page cache; cold: evicted first with
// POSIX_FADV_DONTNEED), D the mapping read as pageable memory where the device says it can. the sum each read returns
// is checked against the CPU's, so a fast wrong read cannot pass.
#include <hip/hip_runtime.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                                    \
    do {                                                                                         \
        hipError_t e_ = (x);                                                                     \
        if (e_ != hipSuccess) {                                                                  \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

__global__ void sum_kernel(const uint4* __restrict__ p, size_t n, unsigned long long* out) {
    unsigned long long s = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint4 v = p[i];
        s += (unsigned long long) v.x + v.y + v.z + v.w;
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor(s, o, 32);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static unsigned long long cpu_sum(const uint8_t* p, size_t bytes) {
    unsigned long long s = 0;
    const uint32_t* w = (const uint32_t*) p;
    for (size_t i = 0; i < bytes / 4; ++i) s += w[i];
    return s;
}

// one read of the whole range by the GPU: GB/s of the median of `reps`, and whether every sum matched
static double read_gbs(const void* dev_ptr, size_t bytes, int reps, unsigned long long want, bool& ok) {
    unsigned long long* d_out = nullptr;
    CK(hipMalloc(&d_out, sizeof(unsigned long long)));
    int sms = 0;
    CK(hipDeviceGetAttribute(&sms, hipDeviceAttributeMultiprocessorCount, 0));
    std::vector<double> t;
    ok = true;
    for (int r = 0; r <= reps; ++r) {
        CK(hipMemset(d_out, 0, sizeof(unsigned long long)));
        CK(hipDeviceSynchronize());
        const double t0 = now_s();
        sum_kernel<<<sms * 16, 256>>>((const uint4*) dev_ptr, bytes / 16, d_out);
        CK(hipDeviceSynchronize());
        const double dt = now_s() - t0;
        unsigned long long got = 0;
        CK(hipMemcpy(&got, d_out, sizeof(got), hipMemcpyDeviceToHost));
        if (got != want) ok = false;
        if (r > 0) t.push_back(dt);   // (the first read: warm-up, not counted)
    }
    CK(hipFree(d_out));
    std::sort(t.begin(), t.end());
    return (double) bytes / t[t.size() / 2] / 1e9;
}

// the cold case on its own: a range nothing has read, evicted before it is mapped (a mapped page cannot be dropped),
// registered (timed: a pinning registration reads the pages here), read once by the GPU (timed: else the GPU's first
// read faults them in), and only then summed by the CPU to check the GPU's sum. also how large a range registers.
static int cold(const char* path, size_t off, size_t bytes) {
    // SPIKE_HOLD_GB: device memory held first (an expert pool's room), to see whether a registration counts against the
    // same limit (ttm's pages_limit)
    void* held = nullptr;
    if (const char* h = std::getenv("SPIKE_HOLD_GB")) {
        const size_t hb = (size_t) (std::atof(h) * (1ull << 30));
        CK(hipMalloc(&held, hb));
        CK(hipMemset(held, 1, hb));
        std::printf("holding %.1f GiB of device memory\n", (double) hb / (1ull << 30));
    }
    const int fd = open(path, O_RDONLY);
    if (fd < 0) { std::perror("open"); return 1; }
    if (posix_fadvise(fd, (off_t) off, (off_t) bytes, POSIX_FADV_DONTNEED) != 0) std::perror("fadvise");
    uint8_t* map = (uint8_t*) mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, (off_t) off);
    if (map == MAP_FAILED) { std::perror("mmap"); return 1; }
    double t0 = now_s();
    const hipError_t er = hipHostRegister(map, bytes, hipHostRegisterMapped | hipHostRegisterReadOnly);
    const double reg = now_s() - t0;
    if (er != hipSuccess) {
        std::printf("cold %.1f GiB: hipHostRegister failed after %.2f s: %s\n", (double) bytes / (1ull << 30), reg,
                    hipGetErrorString(er));
        return 1;
    }
    {   // does the registration count against the GPU's GTT (ttm's pages_limit)? the driver's own count, while it holds
        std::FILE* g = std::fopen("/sys/class/drm/card0/device/mem_info_gtt_used", "r");
        if (!g) g = std::fopen("/sys/class/drm/card1/device/mem_info_gtt_used", "r");
        unsigned long long used = 0;
        if (g && std::fscanf(g, "%llu", &used) == 1)
            std::printf("GTT used while registered: %.2f GiB\n", (double) used / (1ull << 30));
        if (g) std::fclose(g);
    }
    void* dp = nullptr;
    CK(hipHostGetDevicePointer(&dp, map, 0));
    unsigned long long* d_out = nullptr;
    CK(hipMalloc(&d_out, sizeof(unsigned long long)));
    CK(hipMemset(d_out, 0, sizeof(unsigned long long)));
    int sms = 0;
    CK(hipDeviceGetAttribute(&sms, hipDeviceAttributeMultiprocessorCount, 0));
    t0 = now_s();
    sum_kernel<<<sms * 16, 256>>>((const uint4*) dp, bytes / 16, d_out);
    CK(hipDeviceSynchronize());
    const double first = now_s() - t0;
    unsigned long long got = 0;
    CK(hipMemcpy(&got, d_out, sizeof(got), hipMemcpyDeviceToHost));
    bool ok = false;
    const double warm = read_gbs(dp, bytes, 3, got, ok);
    const unsigned long long want = cpu_sum(map, bytes);
    std::printf("cold %.1f GiB: register %.2f s (%.1f GB/s), first GPU read %.2f s (%.1f GB/s), then %.1f GB/s; %s\n",
                (double) bytes / (1ull << 30), reg, (double) bytes / reg / 1e9, first, (double) bytes / first / 1e9,
                warm, got == want && ok ? "sums ok" : "SUM MISMATCH");
    CK(hipHostUnregister(map));
    munmap(map, bytes);
    close(fd);
    if (held) CK(hipFree(held));
    return got == want ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc >= 5 && std::strcmp(argv[4], "cold") == 0)
        return cold(argv[1], (size_t) (std::atof(argv[2]) * (1ull << 30)), (size_t) (std::atof(argv[3]) * (1ull << 30)));
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <file> [offset_gb=8] [size_gb=2] [reps=5]\n", argv[0]);
        return 2;
    }
    const size_t gb = 1ull << 30;
    const size_t off = (size_t) (argc > 2 ? std::atof(argv[2]) * gb : 8 * gb);
    const size_t bytes = (size_t) (argc > 3 ? std::atof(argv[3]) * gb : 2 * gb);
    const int reps = argc > 4 ? std::atoi(argv[4]) : 5;
    const int fd = open(argv[1], O_RDONLY);
    if (fd < 0) {
        std::perror("open");
        return 1;
    }
    uint8_t* map = (uint8_t*) mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, (off_t) off);
    if (map == MAP_FAILED) {
        std::perror("mmap");
        return 1;
    }
    std::printf("file %s, %.1f GiB at %.1f GiB\n", argv[1], (double) bytes / gb, (double) off / gb);
    const unsigned long long want = cpu_sum(map, bytes);   // (also brings the range into the page cache: warm)

    // A: device memory, the file's bytes copied in
    void* dev = nullptr;
    CK(hipMalloc(&dev, bytes));
    CK(hipMemcpy(dev, map, bytes, hipMemcpyHostToDevice));
    bool ok = false;
    double g = read_gbs(dev, bytes, reps, want, ok);
    std::printf("A device memory (hipMalloc)            %7.1f GB/s  %s\n", g, ok ? "sums ok" : "SUM MISMATCH");
    CK(hipFree(dev));

    // B: pinned host memory
    void* pin = nullptr;
    CK(hipHostMalloc(&pin, bytes, hipHostMallocDefault));
    std::memcpy(pin, map, bytes);
    g = read_gbs(pin, bytes, reps, want, ok);
    std::printf("B pinned host memory (hipHostMalloc)   %7.1f GB/s  %s\n", g, ok ? "sums ok" : "SUM MISMATCH");
    CK(hipHostFree(pin));

    // C: the file's pages in place, registered (warm), then evicted and registered again (cold)
    for (int cold = 0; cold < 2; ++cold) {
        if (cold && posix_fadvise(fd, (off_t) off, (off_t) bytes, POSIX_FADV_DONTNEED) != 0) std::perror("fadvise");
        const double t0 = now_s();
        const hipError_t er = hipHostRegister(map, bytes, hipHostRegisterMapped | hipHostRegisterReadOnly);
        const double reg = now_s() - t0;
        if (er != hipSuccess) {
            std::printf("C file pages in place (%s): hipHostRegister failed: %s\n", cold ? "cold" : "warm",
                        hipGetErrorString(er));
            (void) hipGetLastError();
            continue;
        }
        void* dp = nullptr;
        CK(hipHostGetDevicePointer(&dp, map, 0));
        g = read_gbs(dp, bytes, reps, want, ok);
        std::printf("C file pages in place, %s: register %6.2f s (%5.1f GB/s), read %7.1f GB/s  %s\n",
                    cold ? "cold" : "warm", reg, (double) bytes / reg / 1e9, g, ok ? "sums ok" : "SUM MISMATCH");
        CK(hipHostUnregister(map));
    }

    // D: the mapping read as pageable memory, where the device supports it (no registration)
    int pageable = 0;
    CK(hipDeviceGetAttribute(&pageable, hipDeviceAttributePageableMemoryAccess, 0));
    if (pageable) {
        g = read_gbs(map, bytes, reps, want, ok);
        std::printf("D the mapping as pageable memory       %7.1f GB/s  %s\n", g, ok ? "sums ok" : "SUM MISMATCH");
    } else {
        std::printf("D the mapping as pageable memory: the device reports no pageable memory access\n");
    }
    munmap(map, bytes);
    close(fd);
    return 0;
}
