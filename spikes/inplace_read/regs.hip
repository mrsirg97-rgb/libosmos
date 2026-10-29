// spike (design step), after inplace_read: what a registration costs at an expert's size, whether two may share a page,
// and how many may be held at once - the questions a planner that registers experts in place must answer.
//
//   regs <file> [region_mb=3] [count=4000]
//
// 1. cold: `count` regions of `region_mb` at spread offsets, each evicted before it is mapped, registered and then
//    unregistered one at a time: the latency of each call (p50 / p95 / max).
// 2. overlap: two regions sharing a 4 KiB page, both registered, both read by the GPU (sums checked).
// 3. held: `count` warm regions registered and held together until one is refused or all are; one read of the last.
#include <hip/hip_runtime.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                                    \
    do {                                                                                         \
        hipError_t e_ = (x);                                                                     \
        if (e_ != hipSuccess) {                                                                  \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

constexpr size_t kPage = 4096;
constexpr unsigned kRegFlags = hipHostRegisterMapped | hipHostRegisterReadOnly;

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
    for (size_t i = 0; i < bytes / 4; ++i) s += ((const uint32_t*) p)[i];
    return s;
}

static unsigned long long gpu_sum(const void* host, size_t bytes) {
    void* dp = nullptr;
    CK(hipHostGetDevicePointer(&dp, const_cast<void*>(host), 0));
    unsigned long long* d = nullptr;
    CK(hipMalloc(&d, sizeof(*d)));
    CK(hipMemset(d, 0, sizeof(*d)));
    sum_kernel<<<640, 256>>>((const uint4*) dp, bytes / 16, d);
    CK(hipDeviceSynchronize());
    unsigned long long got = 0;
    CK(hipMemcpy(&got, d, sizeof(got), hipMemcpyDeviceToHost));
    CK(hipFree(d));
    return got;
}

static void pct(const char* what, std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const auto at = [&](double p) { return v[std::min(v.size() - 1, (size_t) (p * (double) v.size()))] * 1e3; };
    std::printf("  %-12s p50 %8.3f ms  p95 %8.3f ms  max %8.3f ms  (%zu calls)\n", what, at(0.5), at(0.95), v.back() * 1e3,
                v.size());
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <file> [region_mb=3] [count=4000]\n", argv[0]);
        return 2;
    }
    const size_t region = (size_t) ((argc > 2 ? std::atof(argv[2]) : 3.0) * 1048576.0) / kPage * kPage;
    const int count = argc > 3 ? std::atoi(argv[3]) : 4000;
    const int fd = open(argv[1], O_RDONLY);
    struct stat st {};
    if (fd < 0 || fstat(fd, &st) != 0) {
        std::perror("open");
        return 1;
    }
    const size_t fsize = (size_t) st.st_size / kPage * kPage;
    uint8_t* base = (uint8_t*) mmap(nullptr, fsize, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        std::perror("mmap");
        return 1;
    }
    const size_t stride = std::max(region, fsize / (size_t) count / kPage * kPage);
    std::printf("file %s (%.1f GiB), regions of %.2f MiB, %d of them %.2f MiB apart\n", argv[1], (double) fsize / (1 << 30),
                (double) region / 1048576.0, count, (double) stride / 1048576.0);

    // 1. cold registrations one at a time, at an expert's size
    {
        std::vector<double> reg, unreg;
        const int n = std::min(count, 400);
        for (int i = 0; i < n; ++i) {
            uint8_t* p = base + (size_t) i * stride;
            if (p + region > base + fsize) break;
            madvise(p, region, MADV_DONTNEED);
            posix_fadvise(fd, (off_t) (p - base), (off_t) region, POSIX_FADV_DONTNEED);
            double t0 = now_s();
            CK(hipHostRegister(p, region, kRegFlags));
            reg.push_back(now_s() - t0);
            t0 = now_s();
            CK(hipHostUnregister(p));
            unreg.push_back(now_s() - t0);
        }
        std::printf("1. cold, one at a time:\n");
        pct("register", reg);
        pct("unregister", unreg);
        double tot = 0;
        for (double v : reg) tot += v;
        std::printf("  register throughput %.2f GB/s\n", (double) region * (double) reg.size() / tot / 1e9);
    }

    // 2. two registrations sharing a page
    {
        uint8_t* a = base + 1 * stride;
        uint8_t* b = a + region - kPage;   // b's first page is a's last
        const hipError_t ea = hipHostRegister(a, region, kRegFlags);
        const hipError_t eb = hipHostRegister(b, region, kRegFlags);
        std::printf("2. overlap (one shared page): first %s, second %s", hipGetErrorString(ea), hipGetErrorString(eb));
        if (ea == hipSuccess && eb == hipSuccess) {
            const bool ok = gpu_sum(a, region) == cpu_sum(a, region) && gpu_sum(b, region) == cpu_sum(b, region);
            std::printf(", both read: %s", ok ? "sums ok" : "SUM MISMATCH");
        }
        std::printf("\n");
        (void) hipGetLastError();
        if (eb == hipSuccess) CK(hipHostUnregister(b));
        if (ea == hipSuccess) CK(hipHostUnregister(a));
    }

    // 3. many held at once (warm: the CPU touches them first, so this measures the count, not the SSD)
    {
        std::vector<uint8_t*> held;
        hipError_t last = hipSuccess;
        double t_reg = 0;
        for (int i = 0; i < count; ++i) {
            uint8_t* p = base + (size_t) i * stride;
            if (p + region > base + fsize) break;
            volatile uint8_t sink = 0;
            for (size_t o = 0; o < region; o += kPage) sink ^= p[o];
            (void) sink;
            const double t0 = now_s();
            last = hipHostRegister(p, region, kRegFlags);
            t_reg += now_s() - t0;
            if (last != hipSuccess) break;
            held.push_back(p);
        }
        std::printf("3. held at once: %zu of %d registered (%.1f GiB)%s%s; warm register %.3f ms each\n", held.size(),
                    count, (double) held.size() * (double) region / (1 << 30), last == hipSuccess ? "" : ", refused: ",
                    last == hipSuccess ? "" : hipGetErrorString(last), t_reg / std::max<size_t>(1, held.size()) * 1e3);
        (void) hipGetLastError();
        if (!held.empty()) {
            uint8_t* p = held.back();
            std::printf("   the last one read by the GPU: %s\n", gpu_sum(p, region) == cpu_sum(p, region) ? "sums ok"
                                                                                                      : "SUM MISMATCH");
        }
        const double t0 = now_s();
        for (uint8_t* p : held) CK(hipHostUnregister(p));
        std::printf("   unregistered all in %.2f s\n", now_s() - t0);
    }
    munmap(base, fsize);
    close(fd);
    return 0;
}
