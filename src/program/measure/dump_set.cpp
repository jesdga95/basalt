// basalt/program/measure/dump_set.cpp - see dump_set.hpp.
//
// Lifted out of `main()` unchanged: same reads, same headers, same report lines, same log text.
#include "dump_set.hpp"

#include "basalt/core/layout.hpp"     // ModelGeometry
#include "basalt/core/session.hpp"    // SessionState
#include "basalt/log.hpp"
#include "basalt/program/options.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace basalt::program::measure {

int write_host_dumps(const HostDumpContext& c, std::string& err) {
    const Options& o = *c.o;
    const basalt::core::ModelGeometry& g = *c.g;
    const basalt::core::SessionState& ss = *c.ss;
    (void) err;

    if (!o.dump_mixed.empty()) {
        std::vector<float> mx((size_t) g.n_embd);
        if (cudaMemcpy(mx.data(), ss.block.mixed, mx.size() * sizeof(float), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
            basalt::log::error("basalt generate", "reading mixed back failed");
            return 1;
        }
        std::FILE* mf = std::fopen(o.dump_mixed.c_str(), "wb");
        if (mf == nullptr) {
            basalt::log::error("basalt generate", "cannot write %s", o.dump_mixed.c_str());
            return 1;
        }
        std::fwrite(mx.data(), sizeof(float), mx.size(), mf);
        std::fclose(mf);
        double s2 = 0, mag = 0;
        for (float v : mx) { s2 += (double) v * (double) v; mag += std::fabs((double) v); }
        std::printf("%-24s %s (n_embd %lld, rms %.5g, mean|.| %.5g)\n", "mixed dumped", o.dump_mixed.c_str(),
                    (long long) g.n_embd, std::sqrt(s2 / (double) mx.size()), mag / (double) mx.size());
    }

    // ---- the residual, for bisecting the head against the layers (see `dump_residual`'s note)
    if (!o.dump_residual.empty()) {
        std::vector<float> R((size_t) g.hc * g.n_embd);
        if (cudaMemcpy(R.data(), ss.R, R.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            basalt::log::error("basalt generate", "reading R back failed");
            return 1;
        }
        std::FILE* rf = std::fopen(o.dump_residual.c_str(), "wb");
        if (rf == nullptr) {
            basalt::log::error("basalt generate", "cannot write %s", o.dump_residual.c_str());
            return 1;
        }
        const int32_t hdr[2] = {(int32_t) g.hc, (int32_t) g.n_embd};
        std::fwrite(hdr, sizeof hdr, 1, rf);
        std::fwrite(R.data(), sizeof(float), R.size(), rf);
        std::fclose(rf);
        double mag = 0, mx = 0;
        int bad = 0;
        for (float v : R) {
            if (!std::isfinite(v)) ++bad;
            else { mag += std::fabs((double) v); mx = std::max(mx, (double) std::fabs((double) v)); }
        }
        std::printf("%-24s %s (%d x %lld, nonfinite %d, mean|.| %.4g, max|.| %.4g)\n", "residual dumped",
                    o.dump_residual.c_str(), (int) g.hc, (long long) g.n_embd, bad, mag / (double) R.size(), mx);
    }

    return 0;
}

}  // namespace basalt::program::measure
