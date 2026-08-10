import SwiftUI

// Task #19 (Motion tab redesign + Prompt Lab) — three composable pieces:
// OptimizeBar (enhance-first prompt flow with lineage), VideoTuningPanel
// (#9 Phase 3 Tier A overrides), and PromptLabPanel (trace feed + ratings).

// MARK: - Optimize bar

/// Enhance-first prompt flow: type INTENT, optimize, review/edit the result,
/// render with the attempt id bound (Codex finding #6 — lineage is
/// server-minted, not client-echoed).
struct OptimizeBar: View {
    @Bindable var engine: EngineService
    @Binding var prompt: String
    @Binding var optimizationAttemptId: String?

    @State private var isOptimizing = false
    @State private var outcome: String?
    @State private var templateBadge: String?
    @State private var errorText: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Button {
                    optimize()
                } label: {
                    HStack(spacing: 4) {
                        if isOptimizing { ProgressView().controlSize(.mini) }
                        Image(systemName: "wand.and.stars")
                        Text(isOptimizing ? "Optimizing…" : "Optimize")
                    }
                }
                .disabled(isOptimizing || prompt.trimmingCharacters(in: .whitespaces).isEmpty)
                .help("Rewrite the prompt through the LTX template + local LLM; the render is then attributed to this optimization attempt")

                if let outcome {
                    Text(outcome)
                        .font(.caption2)
                        .padding(.horizontal, 5).padding(.vertical, 1)
                        .background(
                            (outcome == "succeeded" ? Color.green : Color.orange).opacity(0.18),
                            in: Capsule())
                }
                if let templateBadge {
                    Text(templateBadge).font(.caption2).foregroundStyle(.secondary)
                }
                Spacer()
                if optimizationAttemptId != nil {
                    Label("lineage bound", systemImage: "link")
                        .font(.caption2).foregroundStyle(.green)
                }
            }
            if let errorText {
                Text(errorText).font(.caption2).foregroundStyle(.red)
            }
        }
        .onChange(of: prompt) { _, _ in
            // A manual edit after optimizing keeps the lineage: `edited` is
            // derivable server-side because the attempt stores `optimized`.
        }
    }

    private func optimize() {
        isOptimizing = true
        errorText = nil
        Task {
            defer { isOptimizing = false }
            do {
                let result = try await engine.enhancePromptDetailed(prompt, mediaKind: "video")
                prompt = result.prompt
                optimizationAttemptId = result.attemptId
                outcome = result.outcome
                if let tid = result.templateId, let hash = result.templateHash {
                    templateBadge = "\(tid)@\(hash)"
                }
            } catch {
                errorText = "Optimize failed: \(error.localizedDescription)"
            }
        }
    }
}

// MARK: - Tuning panel (#9 Phase 3)

/// Editable Tier A overrides. Only fields the user TOUCHES are sent — the
/// resolved provenance column in the job status shows `request` for exactly
/// those, keeping the effective-config card truthful.
struct VideoTuningPanel: View {
    @Binding var tuning: [String: Any]

    @State private var guidanceRescale: Double = 0
    @State private var useGuidanceRescale = false
    @State private var nagScale: Double = 0
    @State private var useNag = false
    @State private var stgScale: Double = 0
    @State private var useStg = false
    @State private var twoStageOverride = false
    @State private var useTwoStage = false

    var body: some View {
        DisclosureGroup {
            VStack(alignment: .leading, spacing: 8) {
                toggleRow("Guidance rescale", isOn: $useGuidanceRescale) {
                    Slider(value: $guidanceRescale, in: 0...1, step: 0.05)
                    Text(String(format: "%.2f", guidanceRescale)).monospacedDigit().font(.caption)
                }
                toggleRow("NAG scale", isOn: $useNag) {
                    Slider(value: $nagScale, in: 0...20, step: 0.5)
                    Text(String(format: "%.1f", nagScale)).monospacedDigit().font(.caption)
                }
                toggleRow("STG scale", isOn: $useStg) {
                    Slider(value: $stgScale, in: 0...4, step: 0.1)
                    Text(String(format: "%.1f", stgScale)).monospacedDigit().font(.caption)
                }
                toggleRow("Two-stage refine", isOn: $useTwoStage) {
                    Toggle("", isOn: $twoStageOverride).labelsHidden()
                }
                Text("Untouched fields defer to preset → config → env → builtin. The render's resolved_config records where each value came from.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            .padding(.vertical, 4)
            .onChange(of: useGuidanceRescale) { _, _ in rebuild() }
            .onChange(of: guidanceRescale) { _, _ in rebuild() }
            .onChange(of: useNag) { _, _ in rebuild() }
            .onChange(of: nagScale) { _, _ in rebuild() }
            .onChange(of: useStg) { _, _ in rebuild() }
            .onChange(of: stgScale) { _, _ in rebuild() }
            .onChange(of: useTwoStage) { _, _ in rebuild() }
            .onChange(of: twoStageOverride) { _, _ in rebuild() }
        } label: {
            HStack(spacing: 6) {
                Label("Tuning", systemImage: "slider.horizontal.3").font(.caption.weight(.medium))
                if !tuning.isEmpty {
                    Text("\(tuning.count) override\(tuning.count == 1 ? "" : "s")")
                        .font(.caption2)
                        .padding(.horizontal, 5).padding(.vertical, 1)
                        .background(Color.blue.opacity(0.18), in: Capsule())
                }
            }
        }
    }

    @ViewBuilder
    private func toggleRow(
        _ title: String, isOn: Binding<Bool>, @ViewBuilder control: () -> some View
    ) -> some View {
        HStack(spacing: 8) {
            Toggle("", isOn: isOn).labelsHidden().controlSize(.mini)
            Text(title).font(.caption).frame(width: 110, alignment: .leading)
            if isOn.wrappedValue { control() } else {
                Text("inherited").font(.caption2).foregroundStyle(.tertiary)
                Spacer()
            }
        }
    }

    private func rebuild() {
        var t: [String: Any] = [:]
        if useGuidanceRescale { t["guidance_rescale"] = guidanceRescale }
        if useNag { t["nag_scale"] = nagScale }
        if useStg { t["stg_scale"] = stgScale }
        if useTwoStage { t["two_stage"] = twoStageOverride }
        tuning = t
    }
}

// MARK: - Prompt Lab panel

/// The trace feed: what rendered, with what prompt/config, what happened,
/// and a place to attach the human verdict — the rating that makes exports
/// trainable.
struct PromptLabPanel: View {
    @Bindable var engine: EngineService
    @State private var traces: [EngineService.RenderTraceSummary] = []
    @State private var loaded = false

    var body: some View {
        DisclosureGroup {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text(loaded && traces.isEmpty ? "No traces yet — traces appear after the next engine render." : "")
                        .font(.caption2).foregroundStyle(.secondary)
                    Spacer()
                    Button { Task { await refresh() } } label: {
                        Image(systemName: "arrow.clockwise")
                    }.controlSize(.mini)
                }
                ForEach(traces.prefix(20)) { trace in
                    traceRow(trace)
                    Divider()
                }
            }
            .padding(.vertical, 4)
        } label: {
            Label("Prompt Lab", systemImage: "flask").font(.caption.weight(.medium))
        }
        .task { await refresh() }
    }

    private func refresh() async {
        traces = await engine.fetchRenderTraces(limit: 30)
        loaded = true
    }

    @ViewBuilder
    private func traceRow(_ trace: EngineService.RenderTraceSummary) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 6) {
                statusBadge(trace.status)
                Text(trace.submittedAt.map { $0.formatted(date: .omitted, time: .shortened) } ?? "—")
                    .font(.caption2).foregroundStyle(.secondary)
                if trace.optimizationAttemptId != nil {
                    Image(systemName: "link").font(.caption2).foregroundStyle(.green)
                        .help("Optimization lineage recorded")
                }
                Spacer()
                if let rating = trace.rating {
                    Text(rating).font(.caption2).foregroundStyle(.blue)
                    if rating.hasSuffix(":up") {
                        Button { Task { await promote(trace) } } label: {
                            Image(systemName: "star")
                        }.controlSize(.mini).buttonStyle(.borderless)
                            .help("Promote this prompt pair as a few-shot exemplar for the optimizer")
                    }
                } else if trace.status == "succeeded" {
                    Button { Task { await rate(trace, "up") } } label: {
                        Image(systemName: "hand.thumbsup")
                    }.controlSize(.mini).buttonStyle(.borderless)
                    Button { Task { await rate(trace, "down") } } label: {
                        Image(systemName: "hand.thumbsdown")
                    }.controlSize(.mini).buttonStyle(.borderless)
                }
            }
            if let prompt = trace.prompt {
                Text(prompt).font(.caption).lineLimit(2)
            }
            if let error = trace.error {
                Text(error).font(.caption2).foregroundStyle(.red).lineLimit(2)
            }
        }
    }

    private func rate(_ trace: EngineService.RenderTraceSummary, _ vote: String) async {
        await engine.rateRenderTrace(renderId: trace.renderId, vote: vote)
        await refresh()
    }

    private func promote(_ trace: EngineService.RenderTraceSummary) async {
        await engine.promoteTraceExemplar(renderId: trace.renderId)
    }

    @ViewBuilder
    private func statusBadge(_ status: String) -> some View {
        let color: Color = switch status {
        case "succeeded": .green
        case "failed", "abandoned": .red
        case "running": .orange
        default: .gray
        }
        Text(status)
            .font(.caption2)
            .padding(.horizontal, 5).padding(.vertical, 1)
            .background(color.opacity(0.18), in: Capsule())
            .foregroundStyle(color)
    }
}
