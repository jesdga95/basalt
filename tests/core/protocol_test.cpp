// include/basalt/protocol.hpp: protocol lines written from several threads each arrive whole on stdout.
#include "basalt/protocol.hpp"

#include <cstdio>
#include <string>
#include <thread>
#include <vector>

int main() {
    constexpr int kThreads = 4, kLines = 2000;
    std::vector<std::thread> ts;
    for (int t = 0; t < kThreads; ++t)
        ts.emplace_back([t] {
            for (int i = 0; i < kLines; ++i) {
                if (t % 2 == 0) {
                    std::string bt = "BT " + std::to_string(t);
                    for (int e = 0; e < 8; ++e) bt += " " + std::to_string(2570 + i + e);
                    basalt::protocol::emit_text(bt + "\n");
                } else {
                    basalt::protocol::emit("PL %d %d %.0f %.1f\n", i, kLines, 12.0, 3.5);
                }
            }
        });
    for (auto& th : ts) th.join();
    return 0;
}
