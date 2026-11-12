# QFN (Qwen3.8-Flash-Next) targets. Per-file flags (--use_fast_math) are set in CMakeLists.txt by source path.

# The model's CUDA kernels: GDN, gated residual, QSA, heads, router, shared expert, verify, KV, control vectors.
add_library(model_qwen4exp_kernels STATIC
  src/models/qwen4exp/kernels/ngram.cpp
  src/models/qwen4exp/kernels/dec_gdn_rec.cu
  src/models/qwen4exp/kernels/dec_moe.cu
  src/models/qwen4exp/kernels/dec_moe_peer_q0_m0.cu
  src/models/qwen4exp/kernels/dec_moe_peer_q0_m1.cu
  src/models/qwen4exp/kernels/dec_moe_peer_q1_m0.cu
  src/models/qwen4exp/kernels/dec_moe_peer_q1_m1.cu
  src/models/qwen4exp/kernels/s_gemv.cu
  src/models/qwen4exp/kernels/router_top10.cu
  src/models/qwen4exp/kernels/rope.cu
  src/models/qwen4exp/kernels/s2_gemv_q8.cu
  src/models/qwen4exp/kernels/s2_expert_grouped.cu
  src/models/qwen4exp/kernels/shared_expert.cu
  src/models/qwen4exp/kernels/gr.cu
  src/models/qwen4exp/kernels/gdn.cu
  src/models/qwen4exp/kernels/qsa.cu
  src/models/qwen4exp/kernels/native_gr_norm.cu
  src/models/qwen4exp/kernels/native_moe.cu
  src/models/qwen4exp/kernels/native_gdn.cu
  src/models/qwen4exp/kernels/native_gr_postops.cu
  src/models/qwen4exp/kernels/native_router.cu
  src/models/qwen4exp/kernels/native_gdn_preprocess.cu
  src/models/qwen4exp/kernels/native_qsa.cu
  src/models/qwen4exp/kernels/native_rope.cu
  src/models/qwen4exp/kernels/native_ple_postops.cu
  src/models/qwen4exp/kernels/native_qsa_indexer.cu
  src/models/qwen4exp/kernels/ple.cu
  src/models/qwen4exp/kernels/kv_q8.cu
  src/models/qwen4exp/kernels/kv_stream.cu
  src/models/qwen4exp/kernels/kv_q4.cu
  src/models/qwen4exp/kernels/fused_gr.cu
  src/models/qwen4exp/kernels/cvec.cu
  src/models/qwen4exp/kernels/qsa_decode_attn.cu
  src/models/qwen4exp/kernels/qsa_dec_attn.cu
  src/models/qwen4exp/kernels/qsa_dec_fused.cu
  src/models/qwen4exp/kernels/q5k_head.cu
  src/models/qwen4exp/kernels/q6k_head.cu
  src/models/qwen4exp/kernels/q80_proxy.cu
  src/models/qwen4exp/kernels/q80_vhead.cu
  src/models/qwen4exp/kernels/qsa_prompt_attn.cu
  src/models/qwen4exp/kernels/fused_gdn.cu
  src/models/qwen4exp/kernels/qsa_select.cu
  src/models/qwen4exp/kernels/verify_kernels.cu
  src/models/qwen4exp/kernels/dense_tc.cu
  src/models/qwen4exp/kernels/mid_chain.cu
  src/models/qwen4exp/kernels/qsa_mid.cu
  src/models/qwen4exp/kernels/hc_fused.cu
  src/models/qwen4exp/kernels/hc8_gr.cu)
target_link_libraries(model_qwen4exp_kernels PUBLIC kernels)

# The peer window kernels use standard CUDA only (Q8_1/Q2_0 rows, shared-memory rings, doorbells):
# they additionally compile for sm_89 (Ada) so a non-Blackwell card can serve as the expert tier.
# The rest of the kernels (NVFP4 tensor cores, TMA, Blackwell instructions) stay sm_120-only.
set_source_files_properties(
  src/models/qwen4exp/kernels/dec_moe_peer_q0_m0.cu
  src/models/qwen4exp/kernels/dec_moe_peer_q0_m1.cu
  src/models/qwen4exp/kernels/dec_moe_peer_q1_m0.cu
  src/models/qwen4exp/kernels/dec_moe_peer_q1_m1.cu
  PROPERTIES CUDA_ARCHITECTURES "89;120")

# The composed layer: layer and session, the verify windows, the MTP drafter, the conversation state.
add_library(model_qwen4exp_engine STATIC src/models/qwen4exp/decode/layer.cpp src/models/qwen4exp/decode/expert_dispatch.cpp
  src/models/qwen4exp/decode/session.cpp
  src/models/qwen4exp/decode/kv_arena.cpp
  src/models/qwen4exp/decode/layout.cpp
  src/models/qwen4exp/decode/native_head.cpp
  src/models/qwen4exp/decode/native_dense.cpp
  src/models/qwen4exp/decode/verify/batch_capture.cpp
  src/models/qwen4exp/decode/verify/batch_execution.cpp
  src/models/qwen4exp/decode/verify/batch_sessions.cpp
  src/models/qwen4exp/decode/verify/capture.cpp
  src/models/qwen4exp/decode/verify/commit.cpp
  src/models/qwen4exp/decode/verify/cpu_warm.cpp
  src/models/qwen4exp/decode/verify/diagnostics.cpp
  src/models/qwen4exp/decode/verify/execution.cpp
  src/models/qwen4exp/decode/verify/expert_execution.cpp
  src/models/qwen4exp/decode/verify/expert_plan.cpp
  src/models/qwen4exp/decode/verify/expert_resources.cpp
  src/models/qwen4exp/decode/verify/gdn_split.cpp
  src/models/qwen4exp/decode/verify/hc8.cpp
  src/models/qwen4exp/decode/verify/head_split.cpp
  src/models/qwen4exp/decode/verify/lifecycle.cpp
  src/models/qwen4exp/decode/verify/pipeline.cpp
  src/models/qwen4exp/decode/verify/profile.cpp
  src/models/qwen4exp/decode/verify/projection.cpp
  src/models/qwen4exp/decode/verify/record.cpp
  src/models/qwen4exp/decode/verify/record_gdn.cpp
  src/models/qwen4exp/decode/verify/record_head.cpp
  src/models/qwen4exp/decode/verify/record_inputs.cpp
  src/models/qwen4exp/decode/verify/record_layer.cpp
  src/models/qwen4exp/decode/verify/record_moe.cpp
  src/models/qwen4exp/decode/verify/record_qsa.cpp
  src/models/qwen4exp/decode/verify/sampling.cpp
  src/models/qwen4exp/decode/mtp/helpers.cpp
  src/models/qwen4exp/decode/mtp/lifecycle.cpp
  src/models/qwen4exp/decode/mtp/tensors.cpp
  src/models/qwen4exp/decode/mtp/load.cpp
  src/models/qwen4exp/decode/mtp/head.cpp
  src/models/qwen4exp/decode/mtp/forward.cpp
  src/models/qwen4exp/decode/mtp/graphs.cpp
  src/models/qwen4exp/decode/mtp/branches.cpp
  src/models/qwen4exp/decode/mtp/conditional.cpp
  src/models/qwen4exp/decode/mtp/prefill.cpp
  src/models/qwen4exp/decode/mtp/context.cpp
  src/models/qwen4exp/decode/mtp/draft.cpp
  src/models/qwen4exp/decode/mtp/stores.cpp
  src/models/qwen4exp/decode/mtp/batching.cpp
  src/models/qwen4exp/conversation/conversation_snapshot.cpp
  src/models/qwen4exp/conversation/conversation_state.cpp
 
 
  src/core/staged_copy.cpp)
target_include_directories(model_qwen4exp_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(model_qwen4exp_engine PUBLIC core model tiers model_qwen4exp_kernels kernels_cpu ${_gpu_runtime_target})

# The prompt path: batched prompt processing (tensor-core GEMMs through cuBLAS).
add_library(model_qwen4exp_prefill STATIC src/models/qwen4exp/prefill/gemm.cu src/models/qwen4exp/prefill/kernels.cu src/models/qwen4exp/prefill/gdn_chunked.cu
  src/models/qwen4exp/prefill/lifecycle.cpp
  src/models/qwen4exp/prefill/layout.cpp
  src/models/qwen4exp/prefill/projection.cpp
  src/models/qwen4exp/prefill/draft_kv.cpp
  src/models/qwen4exp/prefill/peer.cpp
  src/models/qwen4exp/prefill/pipeline.cpp
  src/models/qwen4exp/prefill/statistics.cpp
  src/models/qwen4exp/prefill/run.cpp
  src/models/qwen4exp/prefill/inputs.cpp
  src/models/qwen4exp/prefill/ple.cpp
  src/models/qwen4exp/prefill/stream.cpp
  src/models/qwen4exp/prefill/ahead.cpp
  src/models/qwen4exp/prefill/layer.cpp
  src/models/qwen4exp/prefill/gdn.cpp
  src/models/qwen4exp/prefill/qsa.cpp
  src/models/qwen4exp/prefill/shared_expert.cpp
  src/models/qwen4exp/prefill/routing.cpp
  src/models/qwen4exp/prefill/cpu_experts.cpp
  src/models/qwen4exp/prefill/experts.cpp
  src/models/qwen4exp/prefill/expert_fused.cpp
  src/models/qwen4exp/prefill/expert_mmq.cpp
  src/models/qwen4exp/prefill/expert_fp4.cpp
  src/models/qwen4exp/prefill/expert_fp16.cpp
  src/models/qwen4exp/prefill/peer_experts.cpp
  src/models/qwen4exp/prefill/expert_stream.cpp
  src/models/qwen4exp/prefill/combine.cpp
  src/models/qwen4exp/prefill/handoff.cpp)
target_include_directories(model_qwen4exp_prefill PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${_gpu_include_directories})
target_link_libraries(model_qwen4exp_prefill PUBLIC model_qwen4exp_engine ${_gpu_blas_target} ${_gpu_runtime_target})
