#include "models.h"

#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"

//
// llama_model_bailing_hybrid (trunk)
//

void llama_model_bailing_hybrid::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,         hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,             hparams.n_lora_q, false);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,            hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,          hparams.n_embd_head_k_mla_impl, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,        hparams.n_embd_head_v_mla_impl, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func);
    ml.get_key(LLM_KV_EXPERT_GROUP_COUNT,                hparams.n_expert_groups, false);
    ml.get_key(LLM_KV_EXPERT_GROUP_USED_COUNT,           hparams.n_group_used, false);
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,              hparams.nextn_predict_layers, false);

    hparams.wkv_head_size = hparams.n_embd / hparams.n_head();

    hparams.n_norm_groups = 4;
    ml.get_key(LLM_KV_ATTENTION_GROUPNORM_GROUPS, hparams.n_norm_groups, false);
    hparams.f_norm_group_eps = hparams.f_norm_rms_eps;

    {
        bool has_per_layer_kv = false;
        for (uint32_t i = 1; i < hparams.n_layer; ++i) {
            if (hparams.n_head_kv((int)i) != hparams.n_head_kv(0)) {
                has_per_layer_kv = true;
                break;
            }
        }

        if (has_per_layer_kv) {
            for (uint32_t i = 0; i < hparams.n_layer; ++i) {
                hparams.recurrent_layer_arr[i] = (hparams.n_head_kv((int)i) == 0);
            }
        } else {
            uint32_t full_attn_interval = 8;
            ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
            for (uint32_t i = 0; i < hparams.n_layer; ++i) {
                hparams.recurrent_layer_arr[i] = ((i + 1) % full_attn_interval != 0);
            }
        }
    }

    switch (hparams.n_layer - hparams.nextn_predict_layers) {
        case 32: type = LLM_TYPE_100B_A6B; break; // Ling-2.6-flash
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_bailing_hybrid::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const bool is_mla = hparams.is_mla();

    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    GGML_ASSERT(n_embd_head_qk_nope >= 1);

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;

    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    GGML_ASSERT(n_expert > 0 && "n_expert must be > 0 for bailing_hybrid");
    GGML_ASSERT(n_expert_used > 0 && "n_expert_used must be > 0 for bailing_hybrid");

    for (int i = 0; i < n_layer; ++i) {
        int flags = 0;
        if (hparams.nextn_predict_layers > 0 && static_cast<uint32_t>(i) >= n_layer - hparams.nextn_predict_layers) {
            flags |= TENSOR_SKIP;
        }

        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);

        if (!hparams.is_recurrent(i)) {
            // Global MLA layer
            if (q_lora_rank > 0) {
                layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, flags);
            }
            layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, flags);

            if (q_lora_rank > 0) {
                layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, flags);
                layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k_mla}, flags);
            } else {
                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_head * n_embd_head_k_mla}, flags);
            }

            layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i),
                {n_embd, kv_lora_rank + n_embd_head_qk_rope}, flags);

            if (is_mla) {
                layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i),
                    {n_embd_head_qk_nope, kv_lora_rank, n_head}, flags);
                layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i),
                    {kv_lora_rank, n_embd_head_v_mla, n_head}, flags);
            } else {
                layer.wkv_b = create_tensor(tn(LLM_TENSOR_ATTN_KV_B, "weight", i),
                    {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v_mla)}, flags);
            }

            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i),
                {n_head * n_embd_head_v_mla, n_embd}, flags);
        } else {
            // Local GLA layer
            layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i),
                {n_embd, 3*n_embd}, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i),
                {n_head * n_embd_head_k, n_embd}, flags);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i),
                {n_embd_head_k}, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i),
                {n_embd_head_k}, flags);

            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i),
                {n_embd, n_embd}, flags);
            layer.layer_out_norm = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i),
                {n_embd}, flags);
        }

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        if (static_cast<uint32_t>(i) < hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i),
                {n_embd, n_ff}, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i),
                {n_ff, n_embd}, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i),
                {n_embd, n_ff}, flags);
        } else {
            const int64_t n_ff_shexp = (hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff_exp) * n_expert_shared;

            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i),
                {n_embd, n_expert}, flags);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i),
                {n_expert}, TENSOR_NOT_REQUIRED | flags);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i),
                {n_embd, n_ff_exp, n_expert}, flags);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i),
                {n_ff_exp, n_embd, n_expert}, flags);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i),
                {n_embd, n_ff_exp, n_expert}, flags);

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i),
                {n_embd, n_ff_shexp}, flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i),
                {n_ff_shexp, n_embd}, flags);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", i),
                {n_embd, n_ff_shexp}, flags);
        }

        // MTP/NextN tensors (loaded but skipped in forward pass)
        if (hparams.nextn_predict_layers > 0 && static_cast<uint32_t>(i) >= n_layer - hparams.nextn_predict_layers) {
            layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i), {2 * n_embd, n_embd}, flags);
            layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM, "weight", i), {n_embd}, flags);
            layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM, "weight", i), {n_embd}, flags);

            // MTP layer has layer_output_norm but is a global (non-recurrent) layer
            if (!hparams.is_recurrent(i)) {
                layer.layer_out_norm = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd}, flags);
            }
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_bailing_hybrid::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_bailing_hybrid::graph::graph(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {

    const bool is_mla = hparams.is_mla();

    const int64_t n_embd_head_k = hparams.n_embd_head_k();
    GGML_UNUSED(n_embd_head_k);

    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    GGML_ASSERT(n_embd_head_qk_nope >= 1);

    const uint32_t kv_lora_rank = hparams.n_lora_kv;

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head_k_mla));

    // GLA linear attention head dimensions
    const int64_t head_dim = n_embd / n_head;
    const int64_t d_inner = n_head * head_dim;
    GGML_ASSERT(d_inner == n_embd);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp = build_inp_mem_hybrid_k();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int  n_transformer_layers = n_layer - hparams.nextn_predict_layers;
    const bool need_mtp_hidden      = hparams.nextn_predict_layers > 0;
    for (int il = 0; il < n_transformer_layers; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (!hparams.is_recurrent(il)) {
            // Global MLA layer
            ggml_tensor * q = NULL;

            const bool is_lite = model.layers[il].wq != nullptr;

            if (!is_lite) {
                q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                cb(q, "q", il);

                q = build_norm(q, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
                cb(q, "q", il);

                q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                cb(q, "q", il);
            } else {
                q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(q, "q", il);
            }

            ggml_tensor * q_nope =
                ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k_mla),
                             ggml_row_size(q->type, n_embd_head_k_mla) * n_head, 0);
            cb(q_nope, "q_nope", il);

            ggml_tensor * q_pe = ggml_view_3d(
                ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k_mla),
                ggml_row_size(q->type, n_embd_head_k_mla) * n_head, ggml_row_size(q->type, n_embd_head_qk_nope));
            cb(q_pe, "q_pe", il);

            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
            cb(kv_cmpr_pe, "kv_cmpr_pe", il);

            ggml_tensor * kv_cmpr =
                ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                             ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            cb(kv_cmpr, "kv_cmpr", il);

            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
            cb(k_pe, "k_pe", il);

            q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(q_pe, "q_pe", il);

            k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(k_pe, "k_pe", il);

            kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(kv_cmpr, "kv_cmpr", il);

            if (is_mla) {
                q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
                cb(q_nope, "q_nope_perm", il);

                ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, model.layers[il].wk_b, q_nope);
                cb(q_nope_absorbed, "q_nope_absorbed", il);

                q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
                cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

                ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
                cb(Qcur, "Qcur", il);

                kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
                cb(kv_cmpr, "kv_cmpr_reshape", il);

                ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = kv_cmpr;
                cb(Vcur, "Vcur", il);

                cur = build_attn(inp->get_attn(),
                        model.layers[il].wo, NULL, model.layers[il].wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, kq_scale, il);
            } else {
                ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_cmpr);
                cb(kv, "kv", il);

                ggml_tensor * k_nope =
                    ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                                 ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v_mla),
                                 ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v_mla) * n_head, 0);
                cb(k_nope, "k_nope_view", il);

                ggml_tensor * Vcur = ggml_view_3d(ctx0, kv, n_embd_head_v_mla, n_head, n_tokens,
                                                  ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v_mla),
                                                  ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v_mla) * n_head,
                                                  ggml_row_size(kv->type, n_embd_head_qk_nope));
                cb(Vcur, "Vcur_view", il);

                Vcur = ggml_cont(ctx0, Vcur);
                cb(Vcur, "Vcur_cont", il);

                ggml_tensor * Qcur = ggml_concat(ctx0, q_nope, q_pe, 0);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = ggml_concat(ctx0, k_nope, ggml_repeat(ctx0, k_pe, q_pe), 0);
                cb(Kcur, "Kcur", il);

                cur = build_attn(inp->get_attn(),
                            model.layers[il].wo, NULL, model.layers[il].wo_s,
                            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            }
        } else {
            // Local GLA layer
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur, n_embd_head_k, n_head, n_head, il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX,
                                 n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX,
                                 n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);

            Qcur = ggml_cont(ctx0, Qcur);
            cb(Qcur, "Qcur", il);
            Kcur = ggml_cont(ctx0, Kcur);
            cb(Kcur, "Kcur", il);
            Vcur = ggml_cont(ctx0, Vcur);
            cb(Vcur, "Vcur", il);

            const float n_h = (float)n_head;
            const float rate = powf(2.0f, -(log2f(n_h) - 3.0f));
            const float denom = std::max(1.0f, (float)(n_transformer_layers - 1));
            const float layer_factor = 1.0f - (float)il / denom + 1e-5f;

            ggml_tensor * h_idx = ggml_arange(ctx0, 1.0f, (float)(n_head + 1), 1.0f);
            ggml_tensor * neg_rate_h = ggml_scale(ctx0, h_idx, -rate);
            ggml_tensor * ln2_neg_rate_h = ggml_scale(ctx0, neg_rate_h, logf(2.0f));
            ggml_tensor * exp_val = ggml_exp(ctx0, ln2_neg_rate_h);
            ggml_tensor * slopes = ggml_scale(ctx0, exp_val, -layer_factor);

            slopes = ggml_exp(ctx0, slopes);

            ggml_tensor * g_slope = ggml_reshape_3d(ctx0, slopes, 1, n_head, 1);
            ggml_tensor * g_full = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd_head_k, n_head, n_tokens);
            g_full = ggml_repeat(ctx0, g_slope, g_full);
            cb(g_full, "g_slope", il);

            const auto * mctx_hybrid = static_cast<const llama_memory_hybrid_context *>(mctx);
            const auto * mctx_recr = mctx_hybrid->get_recr();

            ggml_tensor * gla_state = build_rs(inp->get_recr(), mctx_recr->get_s_l(il), hparams.n_embd_s(), (int32_t)ubatch.n_seqs);

            const int64_t n_seqs_i = (int64_t)ubatch.n_seqs;
            const int64_t S = n_embd_head_k;
            const int64_t state_size_per_seq = S * S * n_head;

            ggml_tensor * gla_output = ggml_gated_linear_attn(ctx0, Kcur, Vcur, Qcur, g_full, gla_state,
                                                                powf(float(n_embd_head_k), -0.5f), false);
            cb(gla_output, "gla_output", il);

            ggml_tensor * gla_out = ggml_view_2d(ctx0, gla_output, d_inner, n_tokens, d_inner * sizeof(float), 0);
            cb(gla_out, "gla_out", il);

            const auto recr_head = mctx_recr->get_head();

            const int64_t state_offset = d_inner * n_tokens;
            ggml_tensor * gla_state_new = ggml_view_1d(ctx0, gla_output, hparams.n_embd_s() * n_seqs_i,
                                                         state_offset * sizeof(float));
            cb(gla_state_new, "gla_state_new", il);

            ggml_build_forward_expand(
                gf, ggml_cpy(ctx0, gla_state_new,
                             ggml_view_1d(ctx0, mctx_recr->get_s_l(il), hparams.n_embd_s() * n_seqs_i,
                                          hparams.n_embd_s() * recr_head * ggml_element_size(mctx_recr->get_s_l(il)))));

            const int n_norm_groups = hparams.n_norm_groups > 0 ? hparams.n_norm_groups : 1;
            const int group_size = d_inner / n_norm_groups;
            gla_out = ggml_reshape_3d(ctx0, gla_out, group_size, n_norm_groups, n_tokens);
            gla_out = ggml_rms_norm(ctx0, gla_out, hparams.f_norm_rms_eps);
            gla_out = ggml_reshape_2d(ctx0, gla_out, d_inner, n_tokens);
            cb(gla_out, "gla_out_normed", il);

            if (model.layers[il].layer_out_norm) {
                gla_out = ggml_mul(ctx0, gla_out, model.layers[il].layer_out_norm);
            }

            ggml_tensor * g_gate = ggml_mul_mat(ctx0, model.layers[il].wqkv_gate, cur);
            cb(g_gate, "g_gate", il);
            g_gate = ggml_sigmoid(ctx0, g_gate);

            cur = ggml_mul(ctx0, gla_out, g_gate);
            cb(cur, "gated_out", il);

            cur = build_lora_mm(model.layers[il].wo, cur);
            cb(cur, "attn_out", il);
        }

        if (il == n_transformer_layers - 1 && inp_out_ids && !need_mtp_hidden) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if (static_cast<uint32_t>(il) < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                model.layers[il].ffn_up, NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            ggml_tensor * gate_inp = model.layers[il].ffn_gate_inp;

            ggml_tensor * moe_out = build_moe_ffn(cur,
                gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, NULL,
                model.layers[il].ffn_gate_shexp, NULL, NULL,
                model.layers[il].ffn_down_shexp, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        if (il == n_transformer_layers - 1 && need_mtp_hidden) {
            cb(cur, "h_pre_norm", -1);
            res->t_h_pre_norm = cur;

            if (inp_out_ids) {
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            }
        }

        inpL = cur;
    }

    cur = inpL;

    if (need_mtp_hidden && res->t_h_pre_norm == nullptr) {
        cb(cur, "h_pre_norm", -1);
        res->t_h_pre_norm = cur;
    }

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

//
// llama_model_bailing_hybrid_mtp (MTP head)
//

void llama_model_bailing_hybrid_mtp::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,         hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,             hparams.n_lora_q, false);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,            hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,          hparams.n_embd_head_k_mla_impl, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,        hparams.n_embd_head_v_mla_impl, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func);
    ml.get_key(LLM_KV_EXPERT_GROUP_COUNT,                hparams.n_expert_groups, false);
    ml.get_key(LLM_KV_EXPERT_GROUP_USED_COUNT,           hparams.n_group_used, false);
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,              hparams.nextn_predict_layers, false);
    GGML_ASSERT(hparams.nextn_predict_layers > 0 && "BAILING_HYBRID_MTP requires nextn_predict_layers > 0");
    GGML_ASSERT(hparams.nextn_predict_layers <= hparams.n_layer);

    hparams.kv_only_nextn = true;
    hparams.n_layer_kv_from_start = -1;
    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        hparams.recurrent_layer_arr[i] = false;
    }

    for (uint32_t i = hparams.n_layer - hparams.nextn_predict_layers; i < hparams.n_layer; ++i) {
        hparams.n_head_kv_arr[i] = 1;
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_bailing_hybrid_mtp::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    GGML_ASSERT(n_embd_head_qk_nope >= 1);

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;

    const int64_t n_ff_exp   = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;
    const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd },          TENSOR_NOT_REQUIRED);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    const uint32_t n_main = n_layer - hparams.nextn_predict_layers;
    for (int i = 0; i < n_layer; ++i) {
        if (static_cast<uint32_t>(i) < n_main) {
            continue;
        }

        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), { n_embd }, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, TENSOR_NOT_REQUIRED);

        if (q_lora_rank > 0) {
            layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
            layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, 0);
            layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k_mla}, 0);
        } else {
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_head * n_embd_head_k_mla}, 0);
        }

        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, 0);
        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i),
            {n_embd, kv_lora_rank + n_embd_head_qk_rope}, 0);

        if (hparams.is_mla()) {
            layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i),
                {n_embd_head_qk_nope, kv_lora_rank, n_head}, 0);
            layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i),
                {kv_lora_rank, n_embd_head_v_mla, n_head}, 0);
        } else {
            layer.wkv_b = create_tensor(tn(LLM_TENSOR_ATTN_KV_B, "weight", i),
                {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v_mla)}, 0);
        }

        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i),
            {n_head * n_embd_head_v_mla, n_embd}, 0);

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), { n_embd, n_expert }, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), { n_ff_exp, n_embd, n_expert }, 0);
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), { n_embd, n_ff_exp, n_expert }, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), { n_embd, n_ff_exp, n_expert }, 0);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), { n_embd, n_ff_shexp }, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), { n_embd, n_ff_shexp }, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), { n_ff_shexp, n_embd }, 0);

        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), { 2 * n_embd, n_embd }, 0);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), { n_embd },              0);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), { n_embd },              0);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", i), { n_embd, n_vocab },     TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), { n_embd, n_vocab },     TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), { n_embd },              TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_bailing_hybrid_mtp::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_bailing_hybrid_mtp::graph::graph(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {

    GGML_ASSERT(hparams.nextn_predict_layers > 0 && "BAILING_HYBRID_MTP requires nextn_predict_layers > 0");
    GGML_ASSERT(hparams.nextn_predict_layers == 1 && "BAILING_HYBRID_MTP currently only supports a single MTP block");

    const bool is_mla = hparams.is_mla();

    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    GGML_ASSERT(n_embd_head_qk_nope >= 1);

    const uint32_t kv_lora_rank = hparams.n_lora_kv;

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head_k_mla));

    const int il = (int) hparams.n_layer - (int) hparams.nextn_predict_layers;
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    auto inp = std::make_unique<llm_graph_input_embd>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->embd);
    ggml_set_name(inp->embd, "mtp_h_input");

    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

    ggml_tensor * h_input  = inp->embd;
    ggml_tensor * tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn_k  =  is_mla ? build_attn_inp_k()  : nullptr;
    auto * inp_attn_kv = !is_mla ? build_attn_inp_kv() : nullptr;

    ggml_tensor * h_norm = build_norm(h_input, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    ggml_tensor * q = NULL;
    if (layer.wq_a && layer.wq_b) {
        q = ggml_mul_mat(ctx0, layer.wq_a, cur);
        cb(q, "mtp_q", il);
        q = build_norm(q, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(q, "mtp_q", il);
        q = ggml_mul_mat(ctx0, layer.wq_b, q);
        cb(q, "mtp_q", il);
    } else if (layer.wq) {
        q = ggml_mul_mat(ctx0, layer.wq, cur);
        cb(q, "mtp_q", il);
    } else {
        q = cur;
    }

    ggml_tensor * q_nope =
        ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k_mla),
                     ggml_row_size(q->type, n_embd_head_k_mla) * n_head, 0);
    cb(q_nope, "mtp_q_nope", il);

    ggml_tensor * q_pe = ggml_view_3d(
        ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k_mla),
        ggml_row_size(q->type, n_embd_head_k_mla) * n_head, ggml_row_size(q->type, n_embd_head_qk_nope));
    cb(q_pe, "mtp_q_pe", il);

    ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    cb(kv_cmpr_pe, "mtp_kv_cmpr_pe", il);

    ggml_tensor * kv_cmpr =
        ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                     ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
    cb(kv_cmpr, "mtp_kv_cmpr", il);

    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                      ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                      ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                      ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
    cb(k_pe, "mtp_k_pe", il);

    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                         ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q_pe, "mtp_q_pe", il);

    k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                         ext_factor, attn_factor, beta_fast, beta_slow);
    cb(k_pe, "mtp_k_pe", il);

    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(kv_cmpr, "mtp_kv_cmpr", il);

    if (is_mla) {
        q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
        cb(q_nope, "mtp_q_nope_perm", il);

        ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
        cb(q_nope_absorbed, "mtp_q_nope_absorbed", il);

        q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
        cb(q_nope_absorbed, "mtp_q_nope_absorbed_perm", il);

        ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
        cb(Qcur, "mtp_Qcur", il);

        kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
        cb(kv_cmpr, "mtp_kv_cmpr_reshape", il);

        ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
        cb(Kcur, "mtp_Kcur", il);

        ggml_tensor * Vcur = kv_cmpr;
        cb(Vcur, "mtp_Vcur", il);

        cur = build_attn(inp_attn_k,
                layer.wo, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, layer.wv_b, kq_scale, il);
        cb(cur, "mtp_attn_out", il);
    } else {
        ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_b, kv_cmpr);
        cb(kv, "mtp_kv", il);

        ggml_tensor * k_nope =
            ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                         ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v_mla),
                         ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v_mla) * n_head, 0);
        cb(k_nope, "mtp_k_nope_view", il);

        ggml_tensor * Vcur = ggml_view_3d(ctx0, kv, n_embd_head_v_mla, n_head, n_tokens,
                                          ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v_mla),
                                          ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v_mla) * n_head,
                                          ggml_row_size(kv->type, n_embd_head_qk_nope));
        cb(Vcur, "mtp_Vcur_view", il);

        Vcur = ggml_cont(ctx0, Vcur);
        cb(Vcur, "mtp_Vcur_cont", il);

        ggml_tensor * Qcur = ggml_concat(ctx0, q_nope, q_pe, 0);
        cb(Qcur, "mtp_Qcur", il);

        ggml_tensor * Kcur = ggml_concat(ctx0, k_nope, ggml_repeat(ctx0, k_pe, q_pe), 0);
        cb(Kcur, "mtp_Kcur", il);

        cur = build_attn(inp_attn_kv,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        cb(cur, "mtp_kqv_out", il);

        if (layer.wo) {
            cur = build_lora_mm(layer.wo, cur);
        }
        cb(cur, "mtp_attn_out", il);
    }

    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "mtp_ffn_inp", il);

    cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    if (static_cast<uint32_t>(il) < hparams.n_layer_dense_lead) {
        cur = build_ffn(cur,
            layer.ffn_up, NULL, NULL,
            layer.ffn_gate, NULL, NULL,
            layer.ffn_down, NULL, NULL,
            NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "mtp_ffn_out", il);
    } else {
        ggml_tensor * moe_out = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il);
        cb(moe_out, "mtp_ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
            layer.ffn_up_shexp, NULL, NULL,
            layer.ffn_gate_shexp, NULL, NULL,
            layer.ffn_down_shexp, NULL, NULL,
            NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "mtp_ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "mtp_ffn_out", il);
    }

    cur = ggml_add(ctx0, cur, ffn_inp);
    cb(cur, "mtp_post_ffn", il);

    res->t_mtp_out = cur;

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "BAILING_HYBRID_MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    GGML_ASSERT(head_w && "BAILING_HYBRID_MTP: missing LM head (nextn.shared_head_head or model.output)");
    cur = build_lora_mm(head_w, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
