#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace basalt::models::inkling2 {

class InklingModel;

// The RAM tier's CPU work shares a window with the card's resident experts.
class CpuSpill {
public:
    ~CpuSpill();
    static std::unique_ptr<CpuSpill> make(const InklingModel& model, std::string& err);
    bool refresh(void* stream);
    const unsigned long long* blobs(int layer) const;
    void start(int layer, int rows, const float* input, const int32_t* ids, void* stream);
    void finish(int layer, int rows, const int32_t* ids, float* output, uint32_t* tally, void* stream);
    bool ok(std::string& err) const;

private:
    struct Impl;
    CpuSpill();
    std::unique_ptr<Impl> impl_;
};

}  // namespace basalt::models::inkling2
