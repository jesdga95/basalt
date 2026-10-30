#pragma once

#include "run.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/layout.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/prefill/gemm.hpp"
#include "basalt/prefill/moe_fused.hpp"
#include "basalt/prefill/moe_fused_iq.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/prefill/moe_fp4.hpp"
#include "basalt/core/peer_experts.hpp"
#include <vector>
#include <future>

namespace basalt::prefill {
using namespace detail;

struct Prefill::Run::Chunk::Experts {
    struct Group;
    Experts(Chunk& chunk, int64_t l) : chunk(chunk), l(l) {}
    bool execute();
    bool fused();
    bool finish();
    void wait_issued(size_t k) { chunk.wait_issued(k); }
    void give_back(size_t upto) { chunk.give_back(upto); }
    void copy_ahead(int64_t L) { chunk.copy_ahead(L); }
    const uint8_t* remote_blob(int64_t l, int32_t e) { return chunk.remote_blob(l,e); }
    Chunk& chunk;
    decltype(Chunk::run)& run = chunk.run;
    decltype(Chunk::m)& m = chunk.m;
    decltype(Chunk::g)& g = chunk.g;
    decltype(Chunk::ss)& ss = chunk.ss;
    decltype(Chunk::t_start)& t_start = chunk.t_start;
    decltype(Chunk::LB)& LB = chunk.LB;
    decltype(Chunk::LE)& LE = chunk.LE;
    decltype(Chunk::remote_device)& remote_device = chunk.remote_device;
    decltype(Chunk::s)& s = chunk.s;
    decltype(Chunk::gdn_floats)& gdn_floats = chunk.gdn_floats;
    decltype(Chunk::prev)& prev = chunk.prev;
    decltype(Chunk::rows_in_place)& rows_in_place = chunk.rows_in_place;
    decltype(Chunk::first_stage)& first_stage = chunk.first_stage;
    decltype(Chunk::single_chunk)& single_chunk = chunk.single_chunk;
    decltype(Chunk::head)& head = chunk.head;
    decltype(Chunk::cs)& cs = chunk.cs;
    decltype(Chunk::ple_on)& ple_on = chunk.ple_on;
    decltype(Chunk::ple_batch)& ple_batch = chunk.ple_batch;
    decltype(Chunk::prev0)& prev0 = chunk.prev0;
    decltype(Chunk::ple_next_err)& ple_next_err = chunk.ple_next_err;
    decltype(Chunk::ple_next)& ple_next = chunk.ple_next;
    decltype(Chunk::ple_buf)& ple_buf = chunk.ple_buf;
    decltype(Chunk::tokens)& tokens = chunk.tokens;
    decltype(Chunk::n)& n = chunk.n;
    decltype(Chunk::pos0)& pos0 = chunk.pos0;
    decltype(Chunk::err)& err = chunk.err;
    decltype(Chunk::owner)& owner = chunk.owner;
    decltype(Chunk::c0)& c0 = chunk.c0;
    decltype(Chunk::T)& T = chunk.T;
    decltype(Chunk::p0)& p0 = chunk.p0;
    decltype(Chunk::ple_pending)& ple_pending = chunk.ple_pending;
    decltype(Chunk::qsa_index)& qsa_index = chunk.qsa_index;
    decltype(Chunk::gdn_index)& gdn_index = chunk.gdn_index;
    decltype(Chunk::lay0)& lay0 = chunk.lay0;
    decltype(Chunk::stream_all)& stream_all = chunk.stream_all;
    decltype(Chunk::ppc)& ppc = chunk.ppc;
    decltype(Chunk::peer_fp4)& peer_fp4 = chunk.peer_fp4;
    decltype(Chunk::seq)& seq = chunk.seq;
    decltype(Chunk::seq_start)& seq_start = chunk.seq_start;
    decltype(Chunk::consumed)& consumed = chunk.consumed;
    decltype(Chunk::a_issued)& a_issued = chunk.a_issued;
    decltype(Chunk::a_consumed)& a_consumed = chunk.a_consumed;
    decltype(Chunk::a_stop)& a_stop = chunk.a_stop;
    decltype(Chunk::iss_ms)& iss_ms = chunk.iss_ms;
    decltype(Chunk::iss_streamed)& iss_streamed = chunk.iss_streamed;
    decltype(Chunk::iss_dma)& iss_dma = chunk.iss_dma;
    decltype(Chunk::issuer)& issuer = chunk.issuer;
    decltype(Chunk::threaded_issue)& threaded_issue = chunk.threaded_issue;
    decltype(Chunk::peer_async)& peer_async = chunk.peer_async;
    decltype(Chunk::ahead_n)& ahead_n = chunk.ahead_n;
    decltype(Chunk::guess_for)& guess_for = chunk.guess_for;
    decltype(Chunk::normed)& normed = chunk.normed;
    int64_t l;
    // The Q2_0 pack's experts on the fused int8 kernels (moe_fused.hpp),
    // grouped on the GPU: no host sync.  Only where every expert's place is known before the routing -
    // the streamed walk, in which every non-resident expert of the layer comes through the ring in id
    // order - and where the MMQ buffers exist: they hold the fused path's own (the per-token int8
    // activations in Xq, the int8 H in H, the grouping tables in GU).  Chunks below stream_all_min()
    // keep MMQ.  A native pack's layer takes the native kernels
    // (moe_fused_iq.hpp) where they cover its two formats, else MMQ (or the FP16 path: IQ1_M).
    const basalt::kernels::cpu::ExpertLayout& lay = basalt::kernels::cpu::expert_layout();
    const bool use_mmq = mmq_plan().any && mmq_plan().layer[(size_t) l];
    const int mmq_gt = lay.native ? lay.fmt[(size_t) l].gu_type : 42;
    const int mmq_dt = lay.native ? lay.fmt[(size_t) l].d_type : 42;
    // NVFP4 experts: FP4 block-scaled tensor cores, read in place from the blobs
    const bool fp4_l = use_mmq && lay.native && mmq_gt == GGML_TYPE_NVFP4 && mmq_dt == GGML_TYPE_NVFP4 &&
                       lay.fmt[(size_t) l].tail != 0 && T >= FP4_MIN_TOKENS;
    const bool fused_nat = use_mmq && stream_all && lay.native && fused::native_supported(mmq_gt, mmq_dt);
    const bool fused_l = (use_mmq && stream_all && !lay.native && fused::enabled()) || fused_nat;
    bool peer_now = false;                // multi-GPU: the peer computed rows of this layer (MMQ path only)
    // CPU share: this layer's CPU experts (rows [T * K - rows_cpu, T * K) of Dm) run on a thread that
    // reads on_cpu, so the future (which waits for the thread) is declared after it
    std::vector<char> on_cpu;
    double cpu_ms = 0;
    std::future<bool> cpu_fut;
    int64_t rows_cpu = 0, n_cpu = 0, n_stream = 0;
};

struct Prefill::Run::Chunk::Experts::Group {
    Group(Experts& experts) : experts(experts) {}
    bool execute();
    bool group();
    bool cpu_download();
    void cpu_select();
    bool cpu_start();
    void prepare_products();
    bool peer_experts();
    bool start_staging();
    const uint8_t* other_card(int32_t e);
    bool stage_one(size_t j);
    void fp4_flush();
    void flush();
    bool compute(size_t j, const uint8_t* blob_dev, int slot);
    bool compute_fp4(size_t j, const uint8_t* blob_dev, int slot);
    bool compute_mmq(size_t j, const uint8_t* blob_dev, int slot);
    bool compute_fp16(size_t j, const uint8_t* blob_dev, int slot);
    bool walk();
    void wait_issued(size_t k) { experts.wait_issued(k); }
    void give_back(size_t upto) { experts.give_back(upto); }
    const uint8_t* remote_blob(int64_t l, int32_t e) { return experts.remote_blob(l,e); }
    Experts& experts;
    decltype(Experts::run)& run = experts.run;
    decltype(Experts::m)& m = experts.m;
    decltype(Experts::g)& g = experts.g;
    decltype(Experts::ss)& ss = experts.ss;
    decltype(Experts::t_start)& t_start = experts.t_start;
    decltype(Experts::LB)& LB = experts.LB;
    decltype(Experts::LE)& LE = experts.LE;
    decltype(Experts::remote_device)& remote_device = experts.remote_device;
    decltype(Experts::s)& s = experts.s;
    decltype(Experts::gdn_floats)& gdn_floats = experts.gdn_floats;
    decltype(Experts::prev)& prev = experts.prev;
    decltype(Experts::rows_in_place)& rows_in_place = experts.rows_in_place;
    decltype(Experts::first_stage)& first_stage = experts.first_stage;
    decltype(Experts::single_chunk)& single_chunk = experts.single_chunk;
    decltype(Experts::head)& head = experts.head;
    decltype(Experts::cs)& cs = experts.cs;
    decltype(Experts::ple_on)& ple_on = experts.ple_on;
    decltype(Experts::ple_batch)& ple_batch = experts.ple_batch;
    decltype(Experts::prev0)& prev0 = experts.prev0;
    decltype(Experts::ple_next_err)& ple_next_err = experts.ple_next_err;
    decltype(Experts::ple_next)& ple_next = experts.ple_next;
    decltype(Experts::ple_buf)& ple_buf = experts.ple_buf;
    decltype(Experts::tokens)& tokens = experts.tokens;
    decltype(Experts::n)& n = experts.n;
    decltype(Experts::pos0)& pos0 = experts.pos0;
    decltype(Experts::err)& err = experts.err;
    decltype(Experts::owner)& owner = experts.owner;
    decltype(Experts::c0)& c0 = experts.c0;
    decltype(Experts::T)& T = experts.T;
    decltype(Experts::p0)& p0 = experts.p0;
    decltype(Experts::ple_pending)& ple_pending = experts.ple_pending;
    decltype(Experts::qsa_index)& qsa_index = experts.qsa_index;
    decltype(Experts::gdn_index)& gdn_index = experts.gdn_index;
    decltype(Experts::lay0)& lay0 = experts.lay0;
    decltype(Experts::stream_all)& stream_all = experts.stream_all;
    decltype(Experts::ppc)& ppc = experts.ppc;
    decltype(Experts::peer_fp4)& peer_fp4 = experts.peer_fp4;
    decltype(Experts::seq)& seq = experts.seq;
    decltype(Experts::seq_start)& seq_start = experts.seq_start;
    decltype(Experts::consumed)& consumed = experts.consumed;
    decltype(Experts::a_issued)& a_issued = experts.a_issued;
    decltype(Experts::a_consumed)& a_consumed = experts.a_consumed;
    decltype(Experts::a_stop)& a_stop = experts.a_stop;
    decltype(Experts::iss_ms)& iss_ms = experts.iss_ms;
    decltype(Experts::iss_streamed)& iss_streamed = experts.iss_streamed;
    decltype(Experts::iss_dma)& iss_dma = experts.iss_dma;
    decltype(Experts::issuer)& issuer = experts.issuer;
    decltype(Experts::threaded_issue)& threaded_issue = experts.threaded_issue;
    decltype(Experts::peer_async)& peer_async = experts.peer_async;
    decltype(Experts::ahead_n)& ahead_n = experts.ahead_n;
    decltype(Experts::guess_for)& guess_for = experts.guess_for;
    decltype(Experts::normed)& normed = experts.normed;
    decltype(Experts::chunk)& chunk = experts.chunk;
    decltype(Experts::l)& l = experts.l;
    decltype(Experts::lay)& lay = experts.lay;
    decltype(Experts::use_mmq)& use_mmq = experts.use_mmq;
    decltype(Experts::mmq_gt)& mmq_gt = experts.mmq_gt;
    decltype(Experts::mmq_dt)& mmq_dt = experts.mmq_dt;
    decltype(Experts::fp4_l)& fp4_l = experts.fp4_l;
    decltype(Experts::fused_nat)& fused_nat = experts.fused_nat;
    decltype(Experts::fused_l)& fused_l = experts.fused_l;
    decltype(Experts::peer_now)& peer_now = experts.peer_now;
    decltype(Experts::on_cpu)& on_cpu = experts.on_cpu;
    decltype(Experts::cpu_ms)& cpu_ms = experts.cpu_ms;
    decltype(Experts::cpu_fut)& cpu_fut = experts.cpu_fut;
    decltype(Experts::rows_cpu)& rows_cpu = experts.rows_cpu;
    decltype(Experts::n_cpu)& n_cpu = experts.n_cpu;
    decltype(Experts::n_stream)& n_stream = experts.n_stream;

    const bool grp_mapped = m.grp_host != nullptr;
    int32_t* ids_h = grp_mapped ? m.grp_host : m.ids_host.data();
    int32_t* slot_h = grp_mapped ? m.grp_host + m.grp_tk : m.slot_host.data();
    int32_t* src_h = grp_mapped ? m.grp_host + 2 * m.grp_tk : m.src_host.data();
    const bool cpu_maybe = owner.cpu_pool_ != nullptr && !stream_all && m.src != nullptr && lay.native &&
                           owner.next_ == nullptr && owner.stage_lb_ == 0;
    const bool pre_mmq = mmq_plan().any && mmq_plan().layer[(size_t) l];
    std::vector<char> on_peer;
    int64_t rows_local = T * K, rows_peer = 0;
    std::vector<int32_t> order, order_peer;
    const size_t mmq_gub = use_mmq ? mmq::matrix_bytes(mmq_gt, 1280, N) : 0;
    const size_t mmq_db = use_mmq ? mmq::matrix_bytes(mmq_dt, N, 640) : 0;
    int stage_next = 0;
    std::vector<int> stage_of, job_of;
    Chunk::StagerDone stager_done{nullptr};
    // In the streamed walk an MMQ group is gathered in ONE launch, after ONE wait on its last
    // streamed copy (the copy stream is in order), and its ring slots are released by ONE event.  A
    // wait and a record per expert would put a GPU bubble each on the compute stream.  The same bytes into the same group slots.  A ring
    // entry the routing skipped inside an open group first gathers what the group holds so far
    // (`flush`), so no more than a group's entries are ever held back from the issuer.
    const bool group_gather = stream_all && use_mmq && lay.native &&
                              MMQ_GROUP <= mmq::kGatherGroupMax;
    mmq::GatherGroup gg;
    int gg_slots[MMQ_GROUP];
    int gg_nslots = 0;   // ring slots gathered by the next flush
    // the FP4 path's open group: its experts' blobs (read in place until the down product is done)
    const uint8_t* fp4_blob[MMQ_GROUP];
    size_t fp4_j0 = 0;
    int fp4_n = 0;
};

}  // namespace basalt::prefill
