// HC8 refresh updates graph-bound copies in place, including restoration, runtime adapters and rejected inputs.
#include "basalt/core/runtime_adapters.hpp"
#include "basalt/kernels/hc_fused.hpp"
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/kernels/dequant_bf16.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace K = basalt::kernels;
namespace C = basalt::core;
namespace fs = std::filesystem;

namespace basalt::core {
struct WeightTableFixture {
    static void put(WeightTable& table, const std::string& name, const WeightRef& ref) {
        table.table_[name] = ref;
    }
};
}  // namespace basalt::core

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL: %s\n", what); ++failures; }
}

void cuda_ok(cudaError_t error) {
    if (error != cudaSuccess) {
        std::printf("FAIL: CUDA: %s\n", cudaGetErrorString(error));
        std::exit(1);
    }
}

uint16_t bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (uint16_t) (bits >> 16);
}

struct Matrix {
    std::vector<uint16_t> w, s;
    std::vector<int8_t> q;

    Matrix(int rows, int cols, bool replacement)
        : w((size_t) rows * cols), s((size_t) rows * (((cols / 32) + 7) & ~7), 0), q(w.size()) {
        const int gpr = cols / 32, srow = (gpr + 7) & ~7;
        std::mt19937 rng(replacement ? 29 : 11);
        for (int row = 0; row < rows; ++row) {
            for (int group = 0; group < gpr; ++group) {
                const int exponent = (row + group + (replacement ? 1 : 0)) % 3;
                const bool zero = (row + group) % 7 == (replacement ? 1 : 0);
                const float scale = zero ? 0.0f : (0.5f * (1 << exponent));
                s[(size_t) row * srow + group] = zero ? 0 : (uint16_t) (0x3800 + exponent * 0x400);
                for (int i = 0; i < 32; ++i) {
                    // An exact absmax fixes the f16 scale; all products are exactly representable in bf16.
                    const int value = zero ? 0 : (i == 0 ? 127 : (i == 1 ? -127 : (int) (rng() % 253) - 126));
                    const size_t index = (size_t) row * cols + group * 32 + i;
                    q[index] = (int8_t) value;
                    w[index] = bf16(value * scale);
                }
            }
        }
    }
};

void run_case(bool owned, int rows, int cols, int devices, std::vector<uint16_t*>& registered) {
    const int before = failures;
    const Matrix baseline(rows, cols, false), replacement(rows, cols, true);
    const size_t bytes = baseline.w.size() * sizeof(uint16_t);
    uint16_t *canonical = nullptr, *staged = nullptr, *output = nullptr;
    cuda_ok(cudaMalloc((void**) &canonical, bytes));
    // Canonical keys remain registered, so retain their allocations until every case has finished.
    registered.push_back(canonical);
    cuda_ok(cudaMalloc((void**) &staged, bytes));
    cuda_ok(cudaMalloc((void**) &output, bytes));
    cuda_ok(cudaMemcpy(canonical, baseline.w.data(), bytes, cudaMemcpyHostToDevice));
    const void* key = canonical;
    if (owned) key = K::hc8_adopt(canonical, rows, cols);
    else K::hc8_prepare(canonical, rows, cols);
    K::Hc8View original;
    if (key == nullptr || !K::hc8_view(key, original)) {
        std::printf("FAIL: HC8 copy registration\n");
        std::exit(1);
    }
    expect(original.owned == owned, "copy ownership");
    const size_t allocated = K::hc8_bytes();

    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cuda_ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    K::hc8_dequant_bf16(original, 0, rows, output, stream);
    cuda_ok(cudaStreamEndCapture(stream, &graph));
    cuda_ok(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));

    auto verify = [&](const Matrix& expected) {
        K::Hc8View current;
        expect(K::hc8_view(key, current), "registered key survives refresh");
        expect(current.q == original.q && current.s == original.s && current.bytes == original.bytes &&
                   current.rows == original.rows && current.cols == original.cols && current.srow == original.srow &&
                   current.owned == original.owned,
               "graph-facing addresses, shape and ownership remain stable");
        expect(K::hc8_bytes() == allocated, "refresh does not allocate another copy");
        std::vector<int8_t> q(expected.q.size());
        std::vector<uint16_t> s(expected.s.size()), result(expected.w.size());
        cuda_ok(cudaMemcpy(q.data(), original.q, q.size(), cudaMemcpyDeviceToHost));
        cuda_ok(cudaMemcpy(s.data(), original.s, s.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        expect(q == expected.q, "quantized values match exact CPU reference");
        expect(s == expected.s, "scales and row padding match exact CPU reference");
        cuda_ok(cudaGraphLaunch(executable, stream));
        cuda_ok(cudaStreamSynchronize(stream));
        cuda_ok(cudaMemcpy(result.data(), output, bytes, cudaMemcpyDeviceToHost));
        expect(result == expected.w, "original graph reads current quantized weights");
    };

    verify(baseline);
    cuda_ok(cudaMemcpy(staged, replacement.w.data(), bytes, cudaMemcpyHostToDevice));
    cuda_ok(cudaMemcpy(canonical, staged, bytes, cudaMemcpyDeviceToDevice));
    // Replacing only the bf16 storage leaves both the HC8 values and the captured reader stale.
    verify(baseline);
    expect(K::hc8_refresh(key, staged, rows, cols), "refresh replacement from separate staging buffer");
    verify(replacement);

    expect(!K::hc8_refresh(nullptr, staged, rows, cols), "reject null key");
    expect(!K::hc8_refresh(staged, staged, rows, cols), "reject unregistered device key");
    expect(!K::hc8_refresh(key, nullptr, rows, cols), "reject null staging buffer");
    expect(!K::hc8_refresh(key, baseline.w.data(), rows, cols), "reject host staging buffer");
    expect(!K::hc8_refresh(key, staged, 0, cols), "reject empty shape");
    expect(!K::hc8_refresh(key, staged, -1, cols), "reject negative shape");
    expect(!K::hc8_refresh(key, staged, rows + 1, cols), "reject row mismatch");
    expect(!K::hc8_refresh(key, staged, rows, cols + 32), "reject aligned column mismatch");
    expect(!K::hc8_refresh(key, staged, rows, cols - 1), "reject unaligned columns");
    if (devices > 1) {
        cuda_ok(cudaSetDevice(1));
        expect(!K::hc8_refresh(key, staged, rows, cols), "reject copy on another current device");
        uint16_t* foreign = nullptr;
        cuda_ok(cudaMalloc((void**) &foreign, bytes));
        cuda_ok(cudaSetDevice(0));
        expect(!K::hc8_refresh(key, foreign, rows, cols), "reject staging buffer on another device");
        cuda_ok(cudaSetDevice(1));
        cuda_ok(cudaFree(foreign));
        cuda_ok(cudaSetDevice(0));
    }
    verify(replacement);

    cuda_ok(cudaMemcpy(canonical, baseline.w.data(), bytes, cudaMemcpyHostToDevice));
    expect(K::hc8_refresh(key, canonical, rows, cols), "restore baseline from canonical bf16 storage");
    verify(baseline);
    if (rows == 320 && cols == 10240) {
        cudaEvent_t start, stop;
        cuda_ok(cudaEventCreate(&start));
        cuda_ok(cudaEventCreate(&stop));
        cuda_ok(cudaEventRecord(start));
        constexpr int repeats = 128;
        for (int i = 0; i < repeats; ++i)
            expect(K::hc8_refresh(key, canonical, rows, cols), "timed refresh succeeds");
        cuda_ok(cudaEventRecord(stop));
        cuda_ok(cudaEventSynchronize(stop));
        float ms = 0;
        cuda_ok(cudaEventElapsedTime(&ms, start, stop));
        const double us = ms * 1000.0 / repeats;
        const size_t traffic = bytes + (size_t) rows * cols + (size_t) rows * original.srow * sizeof(uint16_t);
        std::printf("hc8_refresh %s %dx%d: %.3f us/call, %.1f GB/s\n",
                    owned ? "owned" : "canonical", rows, cols, us, traffic / (us * 1000.0));
        cuda_ok(cudaEventDestroy(start));
        cuda_ok(cudaEventDestroy(stop));
    }
    cuda_ok(cudaGraphExecDestroy(executable));
    cuda_ok(cudaGraphDestroy(graph));
    cuda_ok(cudaStreamDestroy(stream));
    cuda_ok(cudaFree(output));
    cuda_ok(cudaFree(staged));
    std::printf("hc8_refresh %s %dx%d: %s\n", owned ? "owned" : "canonical", rows, cols,
                failures == before ? "PASS" : "FAIL");
}

struct Section { std::string name, data; };

std::string write_container(const fs::path& path, const std::vector<Section>& sections) {
    std::vector<char> header((size_t) C::kHeaderBytes, 0);
    C::FileHeader h{};
    std::memcpy(h.magic, C::kModelMagic, sizeof(h.magic));
    h.version = C::kModelVersion;
    h.n_sections = (uint32_t) sections.size();
    uint64_t end = C::kHeaderBytes;
    std::ofstream out(path, std::ios::binary);
    for (size_t i = 0; i < sections.size(); ++i) {
        C::SectionEntry section{};
        std::memcpy(section.name, sections[i].name.data(), sections[i].name.size());
        section.offset = (end + C::kSectionAlign - 1) / C::kSectionAlign * C::kSectionAlign;
        section.bytes = sections[i].data.size();
        std::memcpy(header.data() + sizeof(h) + i * sizeof(section), &section, sizeof(section));
        out.seekp((std::streamoff) section.offset);
        out.write(sections[i].data.data(), (std::streamsize) section.bytes);
        end = section.offset + section.bytes;
    }
    h.file_bytes = end;
    std::memcpy(header.data(), &h, sizeof(h));
    out.seekp(0);
    out.write(header.data(), (std::streamsize) header.size());
    out.close();
    return path.string();
}

void run_runtime_adapters_case(bool owned, uint32_t source_type, int rows, int cols,
                               std::vector<uint16_t*>& registered) {
    const int before = failures;
    const Matrix baseline(rows, cols, false), replacement(rows, cols, true);
    const size_t bytes = baseline.w.size() * sizeof(uint16_t);
    const fs::path dir = fs::temp_directory_path() /
                         ("basalt-hc8-adapters-" +
                          std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const std::string name = "blk.0.hc_attn_down.weight";
    const std::string q8_name = "blk.0.ssm_conv1d.weight";
    const std::string nat_name = "blk.0.attn_q.weight";
    const std::string shape = "[" + std::to_string(cols) + "," + std::to_string(rows) + "]";
    auto encode_hc = [&](const Matrix& matrix) {
        std::string encoded;
        if (source_type == 30)
            return std::string((const char*) matrix.w.data(), bytes);
        if (source_type == 8) {
            const int stride = ((cols / 32) + 7) & ~7;
            for (int row = 0; row < rows; ++row)
                for (int group = 0; group < cols / 32; ++group) {
                    const uint16_t scale = matrix.s[(size_t) row * stride + group];
                    encoded.append((const char*) &scale, sizeof(scale));
                    encoded.append((const char*) matrix.q.data() + (size_t) row * cols + group * 32, 32);
                }
            return encoded;
        }
        for (uint16_t value : matrix.w) {
            const uint32_t bits = (uint32_t) value << 16;
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            if (source_type == 0) encoded.append((const char*) &f, sizeof(f));
            else {
                const uint16_t half = K::f16_from_f32(f);
                encoded.append((const char*) &half, sizeof(half));
            }
        }
        return encoded;
    };
    const std::string base_hc = encode_hc(baseline), rep_hc = encode_hc(replacement);
    const std::string type_name = source_type == 30 ? "BF16" : source_type == 0 ? "F32" :
                                  source_type == 1 ? "F16" : "Q8_0";
    const size_t source_bytes = base_hc.size();
    std::string base_q8(34, '\x02'), rep_q8(34, '\x05');
    base_q8[0] = '\x00'; base_q8[1] = '\x3c';
    rep_q8[0] = '\x00'; rep_q8[1] = '\x40';
    const std::string base_nat(210, '\x11'), rep_nat(210, '\x22');

    auto pad_align = [](std::string& s) {
        const size_t rem = s.size() % (size_t) C::kTensorAlign;
        if (rem != 0) s.resize(s.size() + (size_t) C::kTensorAlign - rem, '\0');
    };
    std::string base_payload = base_hc;
    pad_align(base_payload);
    const size_t q8_off = base_payload.size();
    base_payload += base_q8;
    pad_align(base_payload);
    const size_t nat_off = base_payload.size();
    base_payload += base_nat;

    std::string rep_payload = rep_hc;
    pad_align(rep_payload);
    rep_payload += rep_q8;
    pad_align(rep_payload);
    rep_payload += rep_nat;

    const std::string base_meta =
        R"({"format":"basalt","version":1,"kind":"model","gguf_kv":{},"tensors":{")" + name +
        R"(":{"type":")" + type_name + R"(","type_id":)" + std::to_string(source_type) +
        R"(,"shape":)" + shape + R"(,"bytes":)" + std::to_string(source_bytes) +
        R"(,"offset":0},")" + q8_name +
        R"(":{"type":"Q8_0","type_id":8,"shape":[32,1],"bytes":34,"offset":)" + std::to_string(q8_off) +
        R"(},")" + nat_name +
        R"(":{"type":"Q6_K","type_id":14,"shape":[256,1],"bytes":210,"offset":)" + std::to_string(nat_off) +
        R"(},")" + nat_name +
        R"(.native":{"type":"Q6_K","type_id":14,"shape":[256,1],"bytes":210,"offset":)" + std::to_string(nat_off) +
        R"(}}})";
    const std::string adapter_meta =
        R"({"format":"basalt","version":1,"kind":"adapter","adapter":{"version":1,"entries":{"0":{"target":{"tensor":")" +
        name + R"(","shape":)" + shape +
        R"(,"region":{"kind":"whole"}},"storage":{"kind":"replace","type_id":)" + std::to_string(source_type) +
        R"(,"section":"adapter/data","offset":0,"bytes":)" + std::to_string(source_bytes) +
        R"(}},"1":{"target":{"tensor":")" + q8_name +
        R"(","shape":[32,1],"region":{"kind":"whole"}},"storage":{"kind":"replace","type_id":8,"section":"adapter/data","offset":)" +
        std::to_string(q8_off) + R"(,"bytes":34}},"2":{"target":{"tensor":")" + nat_name +
        R"(","shape":[256,1],"region":{"kind":"whole"}},"storage":{"kind":"replace","type_id":14,"section":"adapter/data","offset":)" +
        std::to_string(nat_off) + R"(,"bytes":210}}}}})";
    const std::string base_path = write_container(dir / "base.basalt", {{"meta", base_meta}, {"gguf", base_payload}});
    const std::string adapter_path =
        write_container(dir / "adapter.basalt", {{"meta", adapter_meta}, {"adapter/data", rep_payload}});

    uint16_t *canonical = nullptr, *output = nullptr;
    uint8_t *q8_dev = nullptr, *nat_dev = nullptr;
    cuda_ok(cudaMalloc((void**) &canonical, bytes));
    cuda_ok(cudaMalloc((void**) &output, bytes));
    cuda_ok(cudaMalloc((void**) &q8_dev, 34));
    cuda_ok(cudaMalloc((void**) &nat_dev, 210));
    cuda_ok(cudaMemcpy(canonical, baseline.w.data(), bytes, cudaMemcpyHostToDevice));

    C::WeightTable wt;
    {
        C::WeightRef ref;
        ref.data = canonical;
        ref.ne0 = cols;
        ref.ne1 = rows;
        ref.elements = (int64_t) cols * rows;
        ref.kind = C::WeightKind::Bf16InF32;
        ref.bytes = bytes;
        ref.resident = true;
        C::WeightTableFixture::put(wt, name, ref);
    }
    {
        C::WeightRef ref;
        ref.native_data = q8_dev;
        ref.native_type = 8;
        ref.resident = false;
        ref.ne0 = 32;
        ref.ne1 = 1;
        ref.elements = 32;
        C::WeightTableFixture::put(wt, q8_name, ref);
    }
    {
        C::WeightRef ref;
        ref.data = nullptr;
        ref.ne0 = 256;
        ref.ne1 = 1;
        ref.elements = 256;
        ref.native_data = nat_dev;
        ref.native_type = 14;
        ref.resident = false;
        C::WeightTableFixture::put(wt, nat_name, ref);
    }
    const void* key = canonical;
    if (owned) {
        key = K::hc8_adopt(canonical, rows, cols);
        expect(key != nullptr && wt.rebind(name, key), "rebind owned HC8 key into WeightTable");
        cuda_ok(cudaFree(canonical));
        canonical = nullptr;
    } else {
        registered.push_back(canonical);
        K::hc8_prepare(canonical, rows, cols);
    }
    K::Hc8View original;
    expect(key != nullptr && K::hc8_view(key, original), "HC8 view registered for runtime adapters");

    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cuda_ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    K::hc8_dequant_bf16(original, 0, rows, output, stream);
    cuda_ok(cudaStreamEndCapture(stream, &graph));
    cuda_ok(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));

    C::ModelFile base_file;
    C::RuntimeAdapters adapters;
    std::string err;
    expect(base_file.open(base_path, err) && base_file.load_meta(err) && base_file.register_gguf_view(err),
           "open base fixture for runtime adapters");
    expect(adapters.open(base_file, {{adapter_path}}, err), "open runtime adapters");
    expect(adapters.bind_weights(wt, C::PackFiles{}, 0, err), "bind HC8 and native weight rows");

    auto verify = [&](const Matrix& expected, char q8_code, float q8_scale, char nat_byte) {
        std::vector<uint16_t> result(expected.w.size());
        cuda_ok(cudaGraphLaunch(executable, stream));
        cuda_ok(cudaStreamSynchronize(stream));
        cuda_ok(cudaMemcpy(result.data(), output, bytes, cudaMemcpyDeviceToHost));
        expect(result == expected.w, "captured graph reads runtime-adapter HC8 weights");
        if (!owned && canonical != nullptr) {
            std::vector<uint16_t> arena(expected.w.size());
            cuda_ok(cudaMemcpy(arena.data(), canonical, bytes, cudaMemcpyDeviceToHost));
            expect(arena == expected.w, "canonical bf16 arena updated alongside HC8 copy");
        }
        std::vector<uint8_t> nat_host(210);
        float* q8_output = nullptr;
        cuda_ok(cudaMalloc((void**) &q8_output, 32 * sizeof(float)));
        K::dequant_f32(8, q8_dev, 0, 1, 32, q8_output, nullptr);
        std::vector<float> q8_values(32);
        cuda_ok(cudaMemcpy(q8_values.data(), q8_output, 32 * sizeof(float), cudaMemcpyDeviceToHost));
        cuda_ok(cudaFree(q8_output));
        cuda_ok(cudaMemcpy(nat_host.data(), nat_dev, 210, cudaMemcpyDeviceToHost));
        expect(q8_values == std::vector<float>(32, (float) q8_code * q8_scale),
               "native Q8_0 replacement preserves scale-first block layout");
        expect(nat_host == std::vector<uint8_t>(210, (uint8_t) nat_byte),
               "native projection bytes updated in place");
    };

    expect(adapters.select(1, err), "select adapter model");
    verify(replacement, '\x05', 2.0f, '\x22');
    expect(adapters.select(0, err), "restore base model");
    verify(baseline, '\x02', 1.0f, '\x11');

    cuda_ok(cudaGraphExecDestroy(executable));
    cuda_ok(cudaGraphDestroy(graph));
    cuda_ok(cudaStreamDestroy(stream));
    cuda_ok(cudaFree(nat_dev));
    cuda_ok(cudaFree(q8_dev));
    cuda_ok(cudaFree(output));
    std::error_code ignored;
    fs::remove_all(dir, ignored);
    std::printf("hc8_refresh runtime_adapters %s type=%u %dx%d: %s\n", owned ? "owned" : "canonical", source_type, rows, cols,
                failures == before ? "PASS" : "FAIL");
}

template<class T>
std::string encoded(std::initializer_list<T> values) {
    return std::string((const char*) values.begin(), values.size() * sizeof(T));
}

void run_conversion_case(uint32_t source_type, C::WeightKind kind, const std::string& payload,
                         const std::string& expected) {
    const size_t elements = expected.size() / (kind == C::WeightKind::F32 ? 4 : 2);
    const std::string shape = "[" + std::to_string(elements) + ",1]";
    const fs::path dir = fs::temp_directory_path() /
                         ("basalt-adapter-conversion-" +
                          std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const std::string name = kind == C::WeightKind::F32 ? "blk.0.ssm_norm.weight" :
                             kind == C::WeightKind::Bf16InF32 ? "blk.0.ffn_gate_inp.weight" :
                                                               "blk.1.ple_conv1d.weight";
    const std::string type_name = source_type == 30 ? "BF16" : source_type == 1 ? "F16" :
                                  source_type == 3 ? "Q4_1" : source_type == 10 ? "Q2_K" : "F32";
    const std::string base_meta =
        R"({"format":"basalt","version":1,"kind":"model","gguf_kv":{},"tensors":{")" + name +
        R"(":{"type":")" + type_name + R"(","type_id":)" + std::to_string(source_type) +
        R"(,"shape":)" + shape + R"(,"bytes":)" + std::to_string(payload.size()) + R"(,"offset":0}}})";
    const std::string adapter_meta =
        R"({"format":"basalt","version":1,"kind":"adapter","adapter":{"version":1,"entries":{"0":{"target":{"tensor":")" +
        name + R"(","shape":)" + shape + R"(,"region":{"kind":"whole"}},"storage":{"kind":"replace","type_id":)" +
        std::to_string(source_type) + R"(,"section":"adapter/data","offset":0,"bytes":)" +
        std::to_string(payload.size()) + R"(}}}}})";
    const std::string base_path =
        write_container(dir / "base.basalt", {{"meta", base_meta}, {"gguf", std::string(payload.size(), '\0')}});
    const std::string adapter_path =
        write_container(dir / "adapter.basalt", {{"meta", adapter_meta}, {"adapter/data", payload}});
    void* device = nullptr;
    cuda_ok(cudaMalloc(&device, expected.size()));
    C::WeightTable weights;
    C::WeightRef ref;
    ref.data = device;
    ref.bytes = expected.size();
    ref.elements = (int64_t) elements;
    ref.ne0 = (int64_t) elements;
    ref.ne1 = 1;
    ref.kind = kind;
    C::WeightTableFixture::put(weights, name, ref);
    C::ModelFile base;
    C::RuntimeAdapters adapters;
    std::string err;
    expect(base.open(base_path, err) && base.load_meta(err) && base.register_gguf_view(err), "open float conversion fixture");
    expect(adapters.open(base, {{adapter_path}}, err), "resolve float conversion fixture");
    expect(adapters.bind_weights(weights, C::PackFiles{}, 0, err), "bind source and engine dtypes");
    expect(adapters.select(1, err), "select source dtype replacement");
    std::string result(expected.size(), '\0');
    cuda_ok(cudaMemcpy(result.data(), device, result.size(), cudaMemcpyDeviceToHost));
    expect(result == expected, "replacement uses source dtype, rounding, and engine width");
    expect(adapters.select(0, err), "restore converted base dtype");
    cuda_ok(cudaMemcpy(result.data(), device, result.size(), cudaMemcpyDeviceToHost));
    expect(result == std::string(expected.size(), '\0'), "base restoration repeats source conversion");
    cuda_ok(cudaFree(device));
    std::error_code ignored;
    fs::remove_all(dir, ignored);
}

void run_quantized_conversion_cases() {
    std::string q4 = encoded<uint16_t>({0x3c00, 0x3800});
    std::vector<float> q4_expected(32);
    for (int i = 0; i < 16; ++i) {
        q4.push_back((char) (i | ((15 - i) << 4)));
        q4_expected[i] = i + 0.5f;
        q4_expected[16 + i] = 15 - i + 0.5f;
    }
    run_conversion_case(3, C::WeightKind::F32, q4,
                        std::string((const char*) q4_expected.data(), q4_expected.size() * sizeof(float)));
    std::string q2(16, '\0');
    for (int i = 0; i < 16; ++i) q2[i] = (char) ((i % 7 + 1) | ((i % 3) << 4));
    q2 += std::string(32, '\xe4') + std::string(32, '\x1b');
    q2 += encoded<uint16_t>({0x3c00, 0x3800});
    std::vector<float> q2_expected;
    for (int half = 0; half < 2; ++half)
        for (int plane = 0; plane < 4; ++plane)
            for (int group = 0; group < 2; ++group) {
                const int scale_group = half * 8 + plane * 2 + group;
                const float value = (scale_group % 7 + 1) * (half ? 3 - plane : plane) -
                                    0.5f * (scale_group % 3);
                q2_expected.insert(q2_expected.end(), 16, value);
            }
    run_conversion_case(10, C::WeightKind::F32, q2,
                        std::string((const char*) q2_expected.data(), q2_expected.size() * sizeof(float)));
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    cuda_ok(cudaSetDevice(0));
    std::vector<uint16_t*> registered;
    for (bool owned : {true, false}) {
        run_case(owned, 3, 96, devices, registered);      // padded scale rows
        run_case(owned, 320, 10240, devices, registered); // model's HC down matrix
        for (uint32_t source_type : {30u, 0u, 1u, 8u})
            run_runtime_adapters_case(owned, source_type, 3, 96, registered);
    }
    run_conversion_case(30, C::WeightKind::F16InF32,
                        encoded<uint16_t>({0x3f80, 0xc020, 0x3eab, 0x0001}),
                        encoded<uint16_t>({0x3c00, 0xc100, 0x3558, 0x0000}));
    run_conversion_case(30, C::WeightKind::F32,
                        encoded<uint16_t>({0x3f80, 0xc020, 0x3eab, 0x0001}),
                        encoded<uint32_t>({0x3f800000, 0xc0200000, 0x3eab0000, 0x00010000}));
    run_conversion_case(1, C::WeightKind::F32,
                        encoded<uint16_t>({0x3c00, 0xc100, 0x3555, 0x0001}),
                        encoded<uint32_t>({0x3f800000, 0xc0200000, 0x3eaaa000, 0x33800000}));
    run_conversion_case(0, C::WeightKind::Bf16InF32,
                        encoded<uint32_t>({0x3f808000, 0x3f818000, 0xc0200000, 0x00000000}),
                        encoded<uint16_t>({0x3f80, 0x3f82, 0xc020, 0x0000}));
    run_quantized_conversion_cases();
    for (uint16_t* ptr : registered) cuda_ok(cudaFree(ptr));
    std::printf("hc8_refresh: %s\n", failures ? "FAILED" : "PASSED");
    return failures ? 1 : 0;
}
