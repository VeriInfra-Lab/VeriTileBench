"""Emit exact source fragments at the Track 1 substitution boundaries."""

import argparse
import ast
from itertools import product
import math
from pathlib import Path
import struct

from export_driver import DRIVER, check_checkout as check_driver, quoted
from install_candidate import (
    AMAX, MODEL, NGRAM, PRIVATE_KERNELS, IMPORT, CALL, QKV_IMPORT, NATIVE_IMPORT,
    QKV_LOOP, GROUPED_LOOP, QKV_VIEW, NATIVE_VIEW, MLP_WEIGHT_CALL, MLP_WEIGHT_GROUPED,
    CE_LOGIT_REUSE_SUBSTITUTIONS, SAMPLED_CE_LOGIT_REUSE_SUBSTITUTIONS,
    ACTIVATION_KERNEL, CE_ACTIVATION_SUBSTITUTIONS, CROSS_ENTROPY, HEAD_IMPORT, HEAD_REFRESH_IMPORT, git,
    SAMPLED, SAMPLED_IMPORT, GATHER_IMPORT, SAMPLED_COPY, GATHER_COPY,
    ANVIL, POLAR, TRANSPOSE, TRANSPOSE_SUBSTITUTIONS, CE_TRANSPOSE_SUBSTITUTIONS, TRAINING_CE_SUBSTITUTIONS,
    CE_PREDICTION_SUBSTITUTIONS, CE_LOGIT_TABLE_SUBSTITUTIONS, SAMPLED_CE_PREDICTION_SUBSTITUTIONS,
    SAMPLED_CE, SAMPLED_CE_IMPORT, DENSIFY_IMPORT, DENSIFY_LAUNCH, DENSIFY_DISPATCH,
    ATTENTION, CONFIG, ROPE, ROPE_SUBSTITUTIONS, STATIONARY_YARN, WIDE_YARN, YARN_SUBSTITUTIONS,
    LERP, TAIL, TAIL_SUBSTITUTIONS, wide_window_sizes,
    VALUE_PULL, VALUE_PULL_SUBSTITUTIONS, NGRAM_TABLE, NGRAM_TABLE_SUBSTITUTIONS, NGRAM_ADAM, ROW_SCATTER,
    VALUE_EMBED, VALUE_EMBED_SUBSTITUTIONS,
    NGRAM_ADAM_SUBSTITUTIONS, ROW_SCATTER_SUBSTITUTIONS,
    ROW_OWNERS, ROW_OWNER_SUBSTITUTIONS, REPLICATED_ADAM, ANVIL_EPILOGUE_SUBSTITUTIONS,
    ROW_PREFETCH, ROW_PREFETCH_SUBSTITUTIONS,
    ANVIL_RAILS_SUBSTITUTIONS, ANVIL_TRACE_SUBSTITUTIONS, ANVIL_FULL_TILE_SUBSTITUTIONS,
    ANVIL_COOPERATIVE_SUBSTITUTIONS,
    FINEWEB_DATA, FINEWEB_DATA_SUBSTITUTIONS,
    SCHEDULE, SCHEDULE_SUBSTITUTIONS,
    STEP_GRAPHS, STEP_GRAPH_SUBSTITUTIONS, DRIVER_SUBSTITUTIONS,
    PINNED_BATCHES, PINNED_BATCH_SUBSTITUTIONS,
)
from verify_candidate import check_checkout


ARTIFACT = Path(__file__).with_name("SourceAnchors.lean")
TRAINING = Path("track_1_short/training.py")
MODEL_IMPORT = "from track_1_short.perf.kernels.ngram_embed import ngram_embedding\n"
NGRAM_START = {
    "reference": "XBLOCK, NUM_WARPS = 1024, 4\n",
    "candidate": "TOKEN_TILE, CHANNEL_TILE, NUM_WARPS = 4, 256, 4\n",
}
NGRAM_AFTER = "\n\nclass _NgramEmbed"
KERNEL_ANCHORS = {
    "mlp_scale_refresh.py": "candidateKernel",
    "mlp_weights_refresh.py": "candidateWeightKernel",
    "lm_head_refresh.py": "candidateHeadKernel",
    "sampled_rows_gather.py": "candidateGatherKernel",
    "byte_transpose.py": "candidateTransposeKernel",
    "bf16_transpose.py": "candidateBf16TransposeKernel",
    "sampled_grad_densify.py": "candidateDensifyKernel",
    "tail_lerp.py": "candidateTailKernel",
    "tail_norm.py": "candidateTailNormKernel",
    "sparse_row_pack.py": "candidateSparseRowKernel",
    "sparse_row_fetch.py": "candidateSparseRowFetchKernel",
    "ngram_moment_replay.py": "candidateNgramMomentKernel",
    "ngram_adam_entries.py": "candidateNgramEntriesKernel",
    "value_adam_scatter.py": "candidateValueAdamKernel",
    "value_embed_accum.py": "candidateValueAccumKernel",
    "replicated_adam_spans.py": "candidateReplicatedSpanKernel",
    "anvil_epilogue.py": "candidateAnvilEpilogueKernel",
    "anvil_trace.py": "candidateAnvilTraceKernel",
    "anvil_rails.py": "candidateAnvilRailsKernel",
    "graph_stage.py": "candidateGraphStageKernel",
    "batch_packet.py": "candidateBatchPacket",
    "ngram_slots.py": "candidateNgramSlots",
    "activation_fp8.py": "candidateActivationKernel",
    "mudd_coefficients.py": "candidateMuddCoefficientsKernel",
    "mudd_backward.py": "candidateMuddBackwardKernel",
}


def raw(value):
    hashes = "#"
    while '"' + hashes in value:
        hashes += "#"
    return "r" + hashes + '"' + value + '"' + hashes


def fp32_bits(value):
    return int.from_bytes(struct.pack("<f", value), "little")


def tail_rates(source):
    scalar_names = {"TAIL_EMA_WINDOW", "TAIL_AVG_WINDOW", "TAIL_AVG_PERIOD", "TAIL_AVG_RATE",
                    "VALUE_EMBED_AVG_WINDOW", "VALUE_EMBED_AVG_RATE", "BANK_BLEND_WINDOW",
                    "BANK_BLEND_PERIOD", "BANK_BLEND_RATE", "TAIL_EMA_BLEND",
                    "BANK_BLEND", "RICHARDSON_W"}
    wanted = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) in scalar_names:
            wanted.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "tail_ema_rate":
            wanted.append(node)
    env = {"math": math}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(TAIL), "exec"), env)
    ema = env["tail_ema_rate"]
    return [
        ("tailEMAWindow", env["TAIL_EMA_WINDOW"]),
        ("tailAverageWindow", env["TAIL_AVG_WINDOW"]),
        ("tailAveragePeriod", env["TAIL_AVG_PERIOD"]),
        ("tailValueWindow", env["VALUE_EMBED_AVG_WINDOW"]),
        ("tailBankWindow", env["BANK_BLEND_WINDOW"]),
        ("tailBankPeriod", env["BANK_BLEND_PERIOD"]),
        ("tailEMAOneRateBits", fp32_bits(ema(1193, 1194))),
        ("tailEMATwoRateBits", fp32_bits(ema(1191, 1194))),
        ("tailAverageRateBits", fp32_bits(env["TAIL_AVG_RATE"])),
        ("tailValueRateBits", fp32_bits(env["VALUE_EMBED_AVG_RATE"])),
        ("tailBankRateBits", fp32_bits(env["BANK_BLEND_RATE"])),
        ("tailShipEMABits", fp32_bits(env["TAIL_EMA_BLEND"])),
        ("tailShipBankBits", fp32_bits(env["BANK_BLEND"])),
        ("tailShipAfterAverageBits", fp32_bits(env["BANK_BLEND"] - env["RICHARDSON_W"])),
    ]


def model_tail(source):
    model = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "GPT")
    refresh = next(n for n in model.body
                   if isinstance(n, ast.FunctionDef) and n.name == "_quantize_mlp_weights_and_scales")
    start = next(n.lineno for n in refresh.body
                 if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "torch.mul")
    return "".join(source.splitlines(keepends=True)[start - 1:refresh.end_lineno])


def ngram_span(source, mode):
    start = source.index(NGRAM_START[mode])
    end = source.index(NGRAM_AFTER, start)
    return source[start:end]


def head_pack_span(source):
    functions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    first = functions["_quantize_lm_head_dual_kernel"].decorator_list[0].lineno
    last = functions["quantize_lm_head_dual"].end_lineno
    return "".join(source.splitlines(keepends=True)[first - 1:last])


def sampled_anchors(source):
    tree = ast.parse(source)
    schedule = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "CANDIDATES_BY_STAGE" for t in n.targets))
    owners = [("CandidateBuilder", "build"), ("SampledSoftmax", "gather")]
    nodes = [schedule]
    for owner, method in owners:
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
        nodes.append(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method))
    lines = source.splitlines(keepends=True)
    return tuple("".join(lines[node.lineno - 1:node.end_lineno]) for node in nodes)


def sampled_inverse_anchors(source):
    init = method_span(source, "SampledSoftmax", "__init__")
    gather = method_span(source, "SampledSoftmax", "_gather_rows")
    arange = [line for line in init.splitlines(keepends=True)
              if "self.arange = torch.arange(" in line]
    positions = [line for line in gather.splitlines(keepends=True) if "inputs.vocab_pos." in line]
    candidate_view = [line for line in gather.splitlines(keepends=True) if "cand = self.candidates[" in line]
    if len(arange) != 1 or len(positions) != 2 or len(candidate_view) != 1:
        raise ValueError("Unrecognized sampled gradient inverse map construction")
    return arange[0], candidate_view[0], "".join(positions)


def method_span(source, class_name, method_name):
    owner = next(node for node in ast.parse(source).body
                 if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in owner.body
                  if isinstance(node, ast.FunctionDef) and node.name == method_name)
    first_line = method.decorator_list[0].lineno if method.decorator_list else method.lineno
    return "".join(source.splitlines(keepends=True)[first_line - 1:method.end_lineno])


def function_span(source, name):
    fn, = (node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name)
    start = fn.decorator_list[0].lineno if fn.decorator_list else fn.lineno
    return "".join(source.splitlines(keepends=True)[start - 1:fn.end_lineno])


def assignment_span(source, name):
    assignment, = (node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
    return "".join(source.splitlines(keepends=True)[assignment.lineno - 1:assignment.end_lineno])


def training_ce_softcap_span(source):
    names = ("SOFTCAP_A", "SOFTCAP_B", "SOFTCAP_C")
    assignment, = (node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Tuple)
                           and tuple(getattr(element, "id", None) for element in target.elts) == names
                           for target in node.targets))
    values = ast.literal_eval(assignment.value)
    if tuple(map(type, values)) != (float, float, float) or values != (23.0, 5.0, 7.5):
        raise ValueError("Pinned CE defaults must match the compiler specialization")
    return "".join(source.splitlines(keepends=True)[assignment.lineno - 1:assignment.end_lineno])


def numeric_division(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return numeric_division(node.left) / numeric_division(node.right)
    raise ValueError("Expected a numeric constant or division")


def lm_head_scales(source):
    head, = (node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Assign)
             and any(ast.unparse(target) == "self.lm_head" for target in node.targets))
    if not isinstance(head.value, ast.Call) or ast.unparse(head.value.func) != "CastedLinearT":
        raise ValueError("Expected the pinned language-model head")
    span = "".join(source.splitlines(keepends=True)[head.lineno - 1:head.end_lineno])
    scales = {kw.arg: numeric_division(kw.value) for kw in head.value.keywords
              if kw.arg in {"x_s", "w_s", "grad_s"}}
    return span, scales


def lm_head_gradient_scale(source):
    span, scales = lm_head_scales(source)
    return span, scales["grad_s"]


def training_adam_defaults(source):
    manager, = (n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "TrainingManager")
    initialize, = (n for n in manager.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    defaults, = (n for n in initialize.body if isinstance(n, ast.Assign)
                 and any(ast.unparse(t) == "self.adam_defaults" for t in n.targets))
    return "".join(source.splitlines(keepends=True)[defaults.lineno - 1:defaults.end_lineno])


def anvil_coefficients(source):
    node, = (node for node in ast.parse(source).body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == "ANVIL_MAPS" for target in node.targets))
    span = "".join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])
    rows = []
    for a, b, c in ast.literal_eval(node.value):
        bits = [int.from_bytes(struct.pack(fmt, x), "little")
                for fmt, x in zip(("<d", "<f", "<f"), (a, b, c), strict=True)]
        rows.append(f"(0x{bits[0]:016x}, 0x{bits[1]:08x}, 0x{bits[2]:08x})")
    return span, ",\n  ".join(rows)


def anvil_frobenius(source):
    node, = (node for node in ast.parse(source).body if isinstance(node, ast.Assign)
             and ast.unparse(node.targets[0]) == "(FROBENIUS_MARGIN, FROBENIUS_EPS)")
    text = "".join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])
    margin, epsilon = map(fp32_bits, ast.literal_eval(node.value))
    return text, margin, epsilon


def anvil_rail_weight(source):
    names = ("RAIL_FAST_BETA", "RAIL_SLOW_BETA", "RAIL_FAST_WEIGHT", "RAIL_ENGAGE_STEP")
    node, = (node for node in ast.parse(source).body if isinstance(node, ast.Assign)
             and isinstance(node.targets[0], ast.Tuple)
             and tuple(ast.unparse(t) for t in node.targets[0].elts) == names)
    text = "".join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])
    _, slow_beta, _, _ = ast.literal_eval(node.value)
    return text, fp32_bits(1.0 - slow_beta)


def value_adam_configuration(source):
    names = {"VALUE_EMBED_LR_MUL", "VALUE_EMBED_BETAS", "VALUE_EMBED_WD_MUL", "VALUE_EMBED_WD_MUL_PERIOD4"}
    constants = {}
    spans = []
    lines = source.splitlines(keepends=True)
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            constants[node.targets[0].id] = ast.literal_eval(node.value)
            spans.append("".join(lines[node.lineno - 1:node.end_lineno]))
        if isinstance(node, ast.FunctionDef) and node.name == "value_embed_betas_and_wd_mul":
            spans.append("".join(lines[node.lineno - 1:node.end_lineno]))
    if constants.keys() != names or len(spans) != 5:
        raise ValueError("Unrecognized value embedding Adam constants and schedule")
    beta1, beta2 = constants["VALUE_EMBED_BETAS"]
    return spans, {
        "valueBeta1": beta1,
        "valueBeta2": beta2,
        "valueRateScale": constants["VALUE_EMBED_LR_MUL"],
        "valueWeightDecayScale": constants["VALUE_EMBED_WD_MUL"],
        "valuePeriod4WeightDecayScale": constants["VALUE_EMBED_WD_MUL_PERIOD4"],
    }


def fineweb_anchors(source):
    tree = ast.parse(source)
    header = next(n for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "HEADER_BYTES" for t in n.targets))
    read_shard = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_load_data_shard")
    bos = next(n for n in tree.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "BOS_ID" for t in n.targets))
    shard = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Shard")
    scheduled = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ScheduledBatches")
    batches = next(n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == "distributed_data_generator")
    max_docs = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "TRAIN_MAX_NUM_DOCS" for t in n.targets))
    cumulative = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cu_seqlens_rows")
    lines = source.splitlines(keepends=True)
    return ("".join(lines[header.lineno - 1:read_shard.end_lineno]),
            "".join(lines[max_docs.lineno - 1:cumulative.end_lineno]),
            *("".join(lines[n.lineno - 1:n.end_lineno]) for n in (bos, shard, batches, scheduled)))


def common(source, parts):
    for old, marker in parts:
        if source.count(old) != 1:
            raise ValueError("Expected a single linker site")
        source = source.replace(old, marker)
    return source


def render(baseline, candidate):
    model = check_checkout(baseline, candidate)
    head_init, head_scales = lm_head_scales(model)
    grad_scale = head_scales["grad_s"]
    candidate_ce = (candidate / CROSS_ENTROPY).read_text()
    scale_assignment, = ast.parse(assignment_span(candidate_ce, "LM_HEAD_GRAD_SCALE")).body
    if numeric_division(scale_assignment.value) != grad_scale:
        raise ValueError("The CE launcher and model must use the same default gradient scale")
    activation_source = (candidate / ACTIVATION_KERNEL).read_text()
    activation_assignment, = ast.parse(assignment_span(activation_source, "ACTIVATION_SCALE")).body
    if numeric_division(activation_assignment.value) != head_scales["x_s"]:
        raise ValueError("The activation launcher and model must use the same default scale")
    check_driver(baseline)
    check_driver(candidate, candidate=True)
    old_ngram = (baseline / NGRAM).read_text()
    new_ngram = (candidate / NGRAM).read_text()
    ref_scale = model_tail(model)
    width_start = model.index("LONG_WINDOW_LAYERS = ")
    width_end = model.index("# Softmax scales", width_start)
    ref_ngram = ngram_span(old_ngram, "reference")
    opt_ngram = ngram_span(new_ngram, "candidate")
    schedule, builder, wait = sampled_anchors((baseline / SAMPLED).read_text())
    arange, candidate_view, inverse = sampled_inverse_anchors((baseline / SAMPLED).read_text())
    data = (baseline / FINEWEB_DATA).read_text()
    read_shard, cumulative, bos, shard, batches, scheduled = fineweb_anchors(data)
    old_gather = common((baseline / SAMPLED).read_text(),
                        ((SAMPLED_IMPORT, "⟪gather import⟫"), (SAMPLED_COPY, "⟪gather rows⟫")))
    new_gather = common((candidate / SAMPLED).read_text(),
                        ((GATHER_IMPORT, "⟪gather import⟫"), (GATHER_COPY, "⟪gather rows⟫")))
    if old_gather != new_gather:
        raise ValueError("The sampled-softmax source differs outside the recognized replacements")
    model_markers = ((MODEL_IMPORT, "⟪common imports⟫"), (ref_scale, "⟪fp8 scalar refresh⟫"),
                     (QKV_IMPORT, "⟪QKV import⟫"), (QKV_LOOP, "⟪QKV loop⟫"),
                     (QKV_VIEW, "⟪QKV bank view⟫"), (MLP_WEIGHT_CALL, "⟪MLP weights⟫"),
                     (HEAD_IMPORT, "⟪LM head refresh⟫"), (WIDE_YARN, "⟪wide YaRN⟫"))
    candidate_markers = ((IMPORT + MODEL_IMPORT, "⟪common imports⟫"), (CALL, "⟪fp8 scalar refresh⟫"),
                         (NATIVE_IMPORT, "⟪QKV import⟫"), (GROUPED_LOOP, "⟪QKV loop⟫"),
                         (NATIVE_VIEW, "⟪QKV bank view⟫"), (MLP_WEIGHT_GROUPED, "⟪MLP weights⟫"),
                         (HEAD_REFRESH_IMPORT, "⟪LM head refresh⟫"), (STATIONARY_YARN, "⟪wide YaRN⟫"))
    if common(model, model_markers) != common((candidate / MODEL).read_text(), candidate_markers):
        raise ValueError("The model differs outside the recognized replacements")
    if common(old_ngram, ((ref_ngram, "⟪ngram layout⟫"),)) != common(new_ngram, ((opt_ngram, "⟪ngram layout⟫"),)):
        raise ValueError("The n-gram wrappers differ outside the layout replacement")
    files = git(baseline, "ls-files", "--", str(DRIVER), "run.sh", "track_1_short", "evals").splitlines()
    runtime_files = [f for f in files if f.endswith((".py", ".cpp", ".cu", ".h", ".cuh", ".sh"))]
    if any(str(path) in runtime_files for path in PRIVATE_KERNELS) or len(set(runtime_files)) != len(runtime_files):
        raise ValueError("Unexpected private module or duplicate path")
    lines = ["import models.NanoGPT.Track1.DriverSource", "",
             "/-! Exact text used by the Track 1 source linker. Generated by export_source_link.py. -/",
             "namespace VeriTile.Bench.Models.NanoGPT.Track1.Driver", ""]
    license_text = git(baseline, "show", "HEAD:LICENSE").strip()
    lines.insert(1, "/-\n" + license_text + "\n-/")
    lines.append("def runtimeFiles : List String := [\n  " + ",\n  ".join(map(quoted, runtime_files)) + "\n]")
    for path in PRIVATE_KERNELS:
        # Carry the exact kernel text into standalone Lean proof snapshots.
        lines.append(f"\ndef {KERNEL_ANCHORS[path.name]} : String := {raw((candidate / path).read_text())}")
    anvil = (baseline / ANVIL).read_text()
    anvil_maps, coefficient_words = anvil_coefficients(anvil)
    frobenius, margin, epsilon = anvil_frobenius(anvil)
    rails, slow_weight = anvil_rail_weight(anvil)
    lines.append(f"\ndef anvilCoefficientWords : List (UInt64 × Nat × Nat) := [\n  {coefficient_words}]")
    lines.append(f"\ndef anvilFrobeniusWords : Nat × Nat := (0x{margin:08x}, 0x{epsilon:08x})")
    lines.append(f"\ndef anvilSlowWeightBits : Nat := 0x{slow_weight:08x}")
    for name, content in (
        ("pinnedDriver", (baseline / DRIVER).read_text()),
        ("pinnedModelLoss", method_span(model, "GPT", "_loss")),
        ("pinnedEvalProjection", method_span((baseline / MODEL.with_name("layers.py")).read_text(),
                                           "CastedLinearT", "forward")),
        ("referenceReplicatedAdam", (baseline / REPLICATED_ADAM).read_text()),
        ("candidateReplicatedAdam", (candidate / REPLICATED_ADAM).read_text()),
        ("pinnedReplicatedAdamKernel", (baseline / REPLICATED_ADAM.parent / "kernels" / REPLICATED_ADAM.name).read_text()),
        ("referenceImport", MODEL_IMPORT),
        ("candidateImport", IMPORT + MODEL_IMPORT),
        ("referenceScaleTail", ref_scale),
        ("candidateScaleTail", CALL),
        ("referenceMLPWeights", MLP_WEIGHT_CALL),
        ("candidateMLPWeights", MLP_WEIGHT_GROUPED),
        ("referenceHeadImport", HEAD_IMPORT),
        ("candidateHeadImport", HEAD_REFRESH_IMPORT),
        ("pinnedHeadPack", head_pack_span((baseline / CROSS_ENTROPY).read_text())),
        ("pinnedCEHeadInit", head_init),
        ("pinnedCELogitGemm", function_span((baseline / CROSS_ENTROPY).read_text(), "_ce_logit_gemm")),
        ("pinnedCEHeadBackward", method_span((baseline / CROSS_ENTROPY).read_text(),
                                              "FusedSoftcappedCrossEntropy", "backward")),
        ("pinnedCEBackward", function_span((baseline / CROSS_ENTROPY).read_text(), "_ce_backward_gemms")),
        ("pinnedTrainingCEKernel", assignment_span((baseline / CROSS_ENTROPY).read_text(), "CE_KERNEL_SOURCE")),
        ("pinnedTrainingCEBlock", assignment_span((baseline / CROSS_ENTROPY).read_text(), "CE_KERNEL_BLOCK_SIZE")),
        ("pinnedTrainingCEVocabulary", assignment_span((baseline / CROSS_ENTROPY).read_text(), "CE_KERNEL_VOCAB_SIZE")),
        ("pinnedTrainingCESoftcap", training_ce_softcap_span((baseline / CROSS_ENTROPY).read_text())),
        ("pinnedTrainingCECompiler", function_span((baseline / CROSS_ENTROPY).read_text(), "compile_ce_kernel")),
        ("pinnedTrainingCELaunch", function_span((baseline / CROSS_ENTROPY).read_text(), "ce_fwd_bwd")),
        ("pinnedTrainingCEForward", method_span((baseline / CROSS_ENTROPY).read_text(),
                                              "FusedSoftcappedCrossEntropy", "forward")),
        ("pinnedCandidateSchedule", schedule),
        ("pinnedCandidateBuilder", builder),
        ("pinnedGatherWait", wait),
        ("pinnedSampledArange", arange),
        ("pinnedCandidateView", candidate_view),
        ("pinnedSampledInverse", inverse),
        ("referenceGatherImport", SAMPLED_IMPORT),
        ("candidateGatherImport", GATHER_IMPORT),
        ("referenceGather", SAMPLED_COPY),
        ("candidateGather", GATHER_COPY),
        ("pinnedSampledCE", (baseline / SAMPLED_CE).read_text()),
        ("pinnedRope", (baseline / ROPE).read_text()),
        ("pinnedAttention", (baseline / ATTENTION).read_text()),
        ("referenceWideYarn", WIDE_YARN),
        ("candidateWideYarn", STATIONARY_YARN),
        ("pinnedTrainingConfig", (baseline / CONFIG).read_text()),
        ("pinnedWindowSchedule", (baseline / SCHEDULE).read_text()),
        ("pinnedStepGraphs", (baseline / STEP_GRAPHS).read_text()),
        ("pinnedBatchStaging", (baseline / PINNED_BATCHES).read_text()),
        ("pinnedAdamDefaults", training_adam_defaults((baseline / TRAINING).read_text())),
        ("pinnedAdvanceSchedule", method_span((baseline / TRAINING).read_text(), "TrainingManager", "advance_schedule")),
        ("pinnedTrainingReset", method_span((baseline / TRAINING).read_text(), "TrainingManager", "reset")),
        ("pinnedOptimizerDispatch", method_span((baseline / TRAINING).read_text(), "TrainingManager", "step_optimizers")),
        ("pinnedWarmupAdamCadence", method_span((baseline / TRAINING).read_text(), "TrainingManager", "is_adam_step")),
        ("pinnedValidationWindow", method_span((baseline / TRAINING).read_text(), "TrainingManager", "apply_final_ws_ext")),
        ("pinnedAttentionWidths", model[width_start:width_end]),
        ("pinnedQueryWidth", method_span(model, "GPT", "attn_qk_dim")),
        ("pinnedValueWidth", method_span(model, "GPT", "attn_v_dim")),
        ("referenceDensifyImport", SAMPLED_CE_IMPORT),
        ("candidateDensifyImport", DENSIFY_IMPORT),
        ("referenceDensifyLaunch", DENSIFY_LAUNCH),
        ("candidateDensifyLaunch", DENSIFY_DISPATCH),
        ("pinnedTranspose", (baseline / TRANSPOSE).read_text()),
        ("pinnedTailAverages", (baseline / TAIL).read_text()),
        ("pinnedTailLerp", (baseline / LERP).read_text()),
        ("pinnedValuePull", (baseline / VALUE_PULL).read_text()),
        ("pinnedValueEmbed", (baseline / VALUE_EMBED).read_text()),
        ("pinnedNgramTable", (baseline / NGRAM_TABLE).read_text()),
        ("pinnedNgramAdam", (baseline / NGRAM_ADAM).read_text()),
        ("pinnedRowScatter", (baseline / ROW_SCATTER).read_text()),
        ("pinnedRowOwners", (baseline / ROW_OWNERS).read_text()),
        ("pinnedRowPrefetch", (baseline / ROW_PREFETCH).read_text()),
        ("pinnedAnvilStep", method_span((baseline / ANVIL).read_text(), "AnvilAndAdam", "step")),
        ("pinnedAnvilCascade", function_span(anvil, "anvil_cascade")),
        ("pinnedAnvilMaps", anvil_maps),
        ("pinnedAnvilFrobenius", frobenius),
        ("pinnedAnvilRails", rails),
        ("referencePolar", (baseline / POLAR).read_text()),
        ("candidatePolar", (candidate / POLAR).read_text()),
        ("pinnedAnvilBankUpdate", function_span((baseline / ANVIL).read_text(), "anvil_bank_update")),
        ("pinnedAnvilEqualizer", method_span((baseline / ANVIL).read_text(), "AnvilAndAdam", "_rail_equalizer")),
        ("pinnedAnvilDecay", method_span((baseline / ANVIL).read_text(), "AnvilAndAdam", "_sign_aligned_decay_update")),
        ("pinnedValueAdamDispatch", method_span((baseline / ANVIL).read_text(), "AnvilAndAdam", "adam_update_shard")),
        ("pinnedValueAdamPrepare", method_span((baseline / ANVIL).read_text(), "AnvilAndAdam", "_adam_update")),
        ("pinnedValueAdamStep", method_span((baseline / ANVIL).read_text(), "AnvilAndAdam", "_adam_update_step")),
        ("pinnedFineWebRead", read_shard),
        ("pinnedFineWebCumulative", cumulative),
        ("pinnedNextMultiple", function_span((baseline / "track_1_short/model/layers.py").read_text(), "next_multiple_of_n")),
        ("pinnedFineWebBOS", bos),
        ("pinnedFineWebShard", shard),
        ("pinnedFineWebBatches", batches),
        ("pinnedFineWebScheduled", scheduled),
        ("pinnedDeferredGathers", (baseline / "track_1_short/perf/deferred_gathers.py").read_text()),
        ("pinnedFp8RefreshGraphs", (baseline / "track_1_short/perf/cuda_graphs/fp8_refresh_graphs.py").read_text()),
        ("pinnedFineWebSplit", function_span(data, "split_attention_segments")),
        ("referenceQKVImport", QKV_IMPORT),
        ("candidateQKVImport", NATIVE_IMPORT),
        ("referenceQKVLoop", QKV_LOOP),
        ("candidateQKVLoop", GROUPED_LOOP),
        ("referenceQKVView", QKV_VIEW),
        ("candidateQKVView", NATIVE_VIEW),
        ("referenceNgram", ref_ngram),
        ("candidateNgram", opt_ngram),
        ("referenceAmax", (baseline / AMAX).read_text()),
        ("candidateAmax", (candidate / AMAX).read_text()),
    ):
        lines.append(f"\ndef {name} : String := {raw(content)}")
    windows = wide_window_sizes((baseline / CONFIG).read_text())
    changes = [(fp32_bits(old), fp32_bits(old / new), fp32_bits(1 - old / new))
               for old, new in product(windows, repeat=2) if old != new]
    rows = ",\n  ".join("(" + ", ".join(f"0x{bits:08x}" for bits in change) + ")" for change in changes)
    lines.append(f"\ndef wideWindowArguments : Array (Nat × Nat × Nat) := #[\n  {rows}\n]")
    lines.append(f"\ndef windowTauBits : Nat := 0x{fp32_bits(2 * math.pi):08x}")
    for name, value in tail_rates((baseline / TAIL).read_text()):
        lines.append(f"\ndef {name} : Nat := {value:#x}" if name.endswith("Bits")
                     else f"\ndef {name} : Nat := {value}")
    ngram_table = ast.parse((baseline / NGRAM_TABLE).read_text())
    period_change = next(node.value for node in ngram_table.body if isinstance(node, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "NGRAM_ADAM_PERIOD4_START" for t in node.targets))
    lines.append(f"\ndef sparsePeriod4Start : Nat := {ast.literal_eval(period_change)}")
    scalar_names = {"NGRAM_ADAM_BETA2": "sparseBeta2",
                    "NGRAM_LR_MUL": "sparseRateScale",
                    "NGRAM_WD_MUL": "sparseWeightDecayScale",
                    "NGRAM_WD_MUL_PERIOD4": "sparsePeriod4WeightDecayScale"}
    for node in ngram_table.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = scalar_names.get(node.targets[0].id)
            if name:
                bits = int.from_bytes(struct.pack("<d", ast.literal_eval(node.value)), "little")
                lines.append(f"\ndef {name} : UInt64 := 0x{bits:016x}")
    spans, values = value_adam_configuration((baseline / TRAINING).read_text())
    lines.append("\ndef pinnedValueAdamConfig : List String := [\n  " +
                 ",\n  ".join(map(raw, spans)) + "\n]")
    for name, value in values.items():
        bits = int.from_bytes(struct.pack("<d", value), "little")
        lines.append(f"\ndef {name} : UInt64 := 0x{bits:016x}")
    head_grad_bits = int.from_bytes(struct.pack("<d", grad_scale), "little")
    lines.append(f"\ndef pinnedCEHeadGradBits : UInt64 := 0x{head_grad_bits:016x}")
    for name, value in (("Activation", head_scales["x_s"]),
                        ("ActivationReciprocal", 1 / head_scales["x_s"]), ("Weight", head_scales["w_s"])):
        bits = int.from_bytes(struct.pack("<d", value), "little")
        lines.append(f"\ndef pinnedCEHead{name}Bits : UInt64 := 0x{bits:016x}")
    for name, replacements in (("rope", ROPE_SUBSTITUTIONS), ("transpose", TRANSPOSE_SUBSTITUTIONS),
                               ("ceTranspose", CE_TRANSPOSE_SUBSTITUTIONS),
                               ("trainingCE", CE_LOGIT_TABLE_SUBSTITUTIONS + TRAINING_CE_SUBSTITUTIONS +
                                CE_PREDICTION_SUBSTITUTIONS),
                               ("sampledCEPrediction", SAMPLED_CE_PREDICTION_SUBSTITUTIONS),
                               ("ceActivation", CE_ACTIVATION_SUBSTITUTIONS),
                               ("ceLogitReuse", CE_LOGIT_REUSE_SUBSTITUTIONS),
                               ("sampledCELogitReuse", SAMPLED_CE_LOGIT_REUSE_SUBSTITUTIONS),
                               ("yarn", YARN_SUBSTITUTIONS), ("tail", TAIL_SUBSTITUTIONS),
                               ("valuePull", VALUE_PULL_SUBSTITUTIONS), ("valueEmbed", VALUE_EMBED_SUBSTITUTIONS),
                               ("ngramTable", NGRAM_TABLE_SUBSTITUTIONS),
                               ("ngramAdam", NGRAM_ADAM_SUBSTITUTIONS),
                               ("rowOwners", ROW_OWNER_SUBSTITUTIONS),
                               ("rowScatter", ROW_SCATTER_SUBSTITUTIONS),
                               ("rowPrefetch", ROW_PREFETCH_SUBSTITUTIONS),
                               ("fineWebData", FINEWEB_DATA_SUBSTITUTIONS),
                               ("schedule", SCHEDULE_SUBSTITUTIONS),
                               ("stepGraph", STEP_GRAPH_SUBSTITUTIONS),
                               ("driver", DRIVER_SUBSTITUTIONS),
                               ("pinnedBatch", PINNED_BATCH_SUBSTITUTIONS),
                               ("anvilEpilogue", ANVIL_EPILOGUE_SUBSTITUTIONS),
                               ("anvilTrace", ANVIL_TRACE_SUBSTITUTIONS),
                               ("anvilFullTile", ANVIL_FULL_TILE_SUBSTITUTIONS),
                               ("anvilRails", ANVIL_RAILS_SUBSTITUTIONS)):
        pairs = ",\n  ".join(f"({raw(new)}, {raw(old)})" for old, new in replacements)
        lines.append(f"\ndef {name}Replacements : List (String × String) := [\n  {pairs}\n]")
    triples = ",\n  ".join(f"({raw(new)}, {raw(old)}, {count})"
                              for old, new, count in reversed(ANVIL_COOPERATIVE_SUBSTITUTIONS))
    lines.append(f"\ndef anvilCooperativeReplacements : List (String × String × Nat) := [\n  {triples}\n]")
    lines += ["", "end VeriTile.Bench.Models.NanoGPT.Track1.Driver", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    text = render(args.baseline, args.candidate)
    if args.generate:
        ARTIFACT.write_text(text)
    if ARTIFACT.read_text() != text:
        raise ValueError("The Lean link anchors do not match the checked-out training source")
