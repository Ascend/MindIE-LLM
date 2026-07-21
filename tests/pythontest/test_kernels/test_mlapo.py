import torch
import torch_npu

torch.npu.config.allow_internal_format = True
experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=[torch_npu.profiler.ExportType.Text, torch_npu.profiler.ExportType.Db],
    profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
    msprof_tx=False,
    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    l2_cache=False,
    op_attr=False,
    data_simplification=False,
    record_op_args=False,
    gc_detect_threshold=None,
)

with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=1),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result"),
    record_shapes=False,
    profile_memory=False,
    with_stack=False,
    with_modules=False,
    with_flops=False,
    experimental_config=experimental_config,
) as prof:
    wdqkv_nd = torch.randint(4, 5, (2112, 7168), dtype=torch.int8).npu()
    wdqkv = torch_npu.npu_format_cast(wdqkv_nd, 29)

    wuq_nd = torch.randint(4, 5, (8 * 192, 1536), dtype=torch.int8).npu()
    wuq = torch_npu.npu_format_cast(wuq_nd, 29)

    # decode_q_nope, k_cache, decode_q_pe, v_cache = torch_npu.npu_mla_process(
    decode_q_nope, k_cache, decode_q_pe, v_cache = torch.ops.mie_ops.npu_mla_process(
        input=torch.randn((6, 7168), dtype=torch.bfloat16).npu(),
        gamma0=torch.randn((7168), dtype=torch.bfloat16).npu(),
        beta0=torch.zeros(7168).to(torch.bfloat16).npu(),
        wdqkv=wdqkv,  # NZ
        descale0=torch.randn((2112,), dtype=torch.float).npu(),
        gamma1=torch.randn((1536), dtype=torch.bfloat16).npu(),
        beta1=torch.zeros(1536).to(torch.bfloat16).npu(),
        wuq=wuq,  # NZ
        descale1=torch.randn((8 * 192,), dtype=torch.float).npu(),
        gamma2=torch.randn((512), dtype=torch.bfloat16).npu(),
        cos=torch.randn((6, 64), dtype=torch.bfloat16).npu(),
        sin=torch.randn((6, 64), dtype=torch.bfloat16).npu(),
        wuk=torch.randn((8, 128, 512), dtype=torch.bfloat16).npu(),
        kv_cache=torch.full([8, 128, 1, 512], float("nan"), dtype=torch.bfloat16).npu(),
        kv_cache_rope=torch.full([8, 128, 1, 64], float("nan"), dtype=torch.bfloat16).npu(),
        slotmapping=torch.tensor([1, 3, 5, 7, 9, 11], dtype=torch.int32).npu(),
        quant_scale0=torch.tensor([1], dtype=torch.bfloat16).npu(),
        quant_offset0=torch.tensor([0], dtype=torch.int8).npu(),
        bias0=torch.randint(4, 5, (1536,), dtype=torch.int32).npu(),
        quant_scale1=torch.tensor([1], dtype=torch.bfloat16).npu(),
        quant_offset1=torch.tensor([0], dtype=torch.int8).npu(),
        bias1=torch.randint(4, 5, (1536,), dtype=torch.int32).npu(),
        ctkv_scale=torch.tensor([1], dtype=torch.bfloat16).npu(),
        q_nope_scale=torch.tensor([1], dtype=torch.bfloat16).npu(),
        cache_mode_opt="krope_ctkv",
        quant_mode_opt="per_tensor_quant_asymm",
    )

    print(f"decode_q_nope is {decode_q_nope} end, {k_cache} ,end {decode_q_pe} end {v_cache} end")
