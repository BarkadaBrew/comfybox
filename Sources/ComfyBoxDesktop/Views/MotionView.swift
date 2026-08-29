// MotionView.swift — LTX-2 video generation (text-to-video & image-to-video)
//
// Calls /v1/video/generate (local LTX-2 backend). A reference image switches
// it to image-to-video. Generation is long (minutes) and synchronous on the
// local backend, so the UI shows a busy state and then an inline AVKit
// preview of the resulting MP4.

import SwiftUI
import AVKit
import AppKit
import UniformTypeIdentifiers

struct MotionView: View {
    @Bindable var engine: EngineService
    /// Reference image queued by Gallery/Canvas "Animate"; consumed once.
    @Binding var pendingMotionReference: String?

    @State private var prompt: String = ""
    @State private var referencePath: String?
    @State private var referenceThumb: NSImage?
    @State private var isReferenceDropTargeted: Bool = false
    @State private var resolution: VideoResolution = .landscape
    @State private var frames: Int = 97
    /// Clip length in SECONDS — the unit you actually think in. LTX renders
    /// 1+8k frames on a 24fps playback basis, so `frames` (still the value the
    /// request carries) is DERIVED from this and snapped to the nearest legal
    /// count. Previously the only control was a fixed frame-count Picker that
    /// merely *displayed* seconds — you could not ask for "6 seconds".
    @State private var seconds: Double = 4.0
    /// Temporal conditioning rate (`tuning.cond_fps`) — the real motion dial:
    /// the same action spread over a LOWER cond_fps reads as MORE movement.
    /// Independent of the 24fps playback basis. 0 = auto (engine picks, which
    /// is the previous behaviour). The engine floors a set value at 6.
    @State private var condFps: Double = 0
    @State private var steps: Double = 8
    @State private var didApplyDefaults = false
    @State private var seedText: String = ""
    // 0.5 is the VALIDATED i2v strength at production config (2026-08-03:
    // 0.75 collapsed long single passes; 1.0 is the historical frame-0
    // ghosting zone). Desktop default matches the daemon recipe.
    @State private var strength: Double = 0.5
    @State private var extendSeconds: Double = 0
    @State private var selectedLoras: [LoRASelection] = []
    @State private var tuningOverrides: [String: Any] = [:]
    @State private var optimizationAttemptId: String?

    // ── Advanced per-render tuning overrides ────────────────────────────────
    // These knobs are otherwise only settable as LTX2_* environment variables
    // in the launchd plist, which needs an edit + engine restart and applies
    // GLOBALLY. Surfacing them here makes them per-render, which is what
    // experimenting actually requires.
    //
    // Sentinel: -1 (or "" for strings) means INHERIT — the key is omitted from
    // the request entirely, so the engine's env/plist default applies exactly
    // as it does today. 0 is a meaningful value for several of these
    // (guidance_rescale, color_anchor, nag_scale all default to 0), which is
    // why the sentinel is -1 and not 0. Ranges mirror the engine's own
    // registry in LTX2ConfigResolver.swift.
    @State private var showAdvanced = false
    @State private var advSampler: String = ""            // "" = inherit
    @State private var advGuidanceRescale: Double = -1    // engine: 0...1
    @State private var advImgCompression: Double = -1     // engine: 0...100 (i2v)
    @State private var advColorAnchor: Double = -1        // engine: 0...1
    @State private var advNagScale: Double = -1           // engine: 0...50
    @State private var advNagAlpha: Double = -1           // engine: 0...1
    @State private var advNagTau: Double = -1             // engine: 1...10
    @State private var advTwoStage: Int = -1              // -1 inherit, 0 off, 1 on

    /// Samplers the LTX-2 pipeline accepts. Empty tag = inherit.
    private static let samplerOptions = ["", "euler", "euler_cfg_pp", "euler_ancestral_cfg_pp", "res_2s"]

    /// Advanced overrides that are actually set (i.e. not inherit), as
    /// snake_case wire keys matching the engine's LTX2VideoTuning. Single
    /// source of truth for BOTH the request body and the on-screen summary, so
    /// the badge can never claim something the render does not send.
    private var advancedOverrides: [String: Any] {
        var t: [String: Any] = [:]
        if !advSampler.isEmpty { t["sampler"] = advSampler }
        if advGuidanceRescale >= 0 { t["guidance_rescale"] = advGuidanceRescale }
        if advImgCompression >= 0 { t["img_compression"] = Int(advImgCompression) }
        if advColorAnchor >= 0 { t["color_anchor"] = advColorAnchor }
        if advNagScale >= 0 { t["nag_scale"] = advNagScale }
        if advNagAlpha >= 0 { t["nag_alpha"] = advNagAlpha }
        if advNagTau >= 1 { t["nag_tau"] = advNagTau }   // engine range starts at 1
        if advTwoStage >= 0 { t["two_stage"] = advTwoStage == 1 }
        return t
    }

    /// Comma-joined key list for the "Overriding: …" badge.
    private var activeOverrideSummary: String {
        advancedOverrides.keys.sorted().joined(separator: ", ")
    }

    @State private var isGenerating = false
    @State private var statusMessage: String?
    @State private var errorMessage: String?
    @State private var resultURL: URL?
    /// Owned here, built once when a result lands — never inline in the view
    /// body. See SafeVideoPlayer.swift / issue #257.
    @State private var player: AVPlayer?

    enum VideoResolution: String, CaseIterable, Identifiable {
        case landscape = "704 × 448 (16:10)"
        case portrait = "448 × 704 (10:16)"
        case square = "512 × 512"
        case wide = "768 × 448 (16:9-ish)"
        var id: String { rawValue }
        var size: (Int, Int) {
            switch self {
            case .landscape: return (704, 448)
            case .portrait: return (448, 704)
            case .square: return (512, 512)
            case .wide: return (768, 448)
            }
        }
    }

    /// LTX-2 requires 1 + 8k frames; durations at 24fps. Up to 289f (12s)
    /// renders SINGLE-PASS — the old 97f ceiling was never real (2026-08-02:
    /// one 193f pass beat 97+97 chaining, 2x faster, no seam).
    private static let frameOptions = [25, 49, 97, 121, 145, 193, 241, 289]

    /// Playback basis LTX renders against. Clip length = frames / 24. This is
    /// NOT the motion dial — that is `cond_fps` (see `condFps`).
    private static let playbackFps: Double = 24

    /// Legal frame counts are 1+8k, from 25f (~1s) to the 289f (~12s)
    /// single-pass cap. Snap a seconds request to the nearest one.
    private static func framesForSeconds(_ s: Double) -> Int {
        let k = ((s * playbackFps) - 1) / 8
        return min(36, max(3, Int(k.rounded()))) * 8 + 1
    }

    private static func secondsForFrames(_ f: Int) -> Double {
        (Double(f) / playbackFps * 10).rounded() / 10
    }

    var body: some View {
        HSplitView {
            controls.frame(minWidth: 320, maxWidth: 420)
            preview.frame(minWidth: 420)
        }
        .navigationTitle("Motion")
        .onAppear { applyDefaults(); consumePendingReference() }
        .onChange(of: pendingMotionReference) { _, _ in consumePendingReference() }
    }

    // MARK: - Controls

    private var controls: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 8) {
                    Image(systemName: "film.stack").foregroundStyle(.purple)
                    Text("LTX-2 Video").font(.headline)
                    Spacer()
                    if !engine.connectionState.isConnected {
                        Text("offline").font(.caption2).foregroundStyle(.orange)
                    }
                }

                labeled("Prompt") {
                    TextEditor(text: $prompt)
                        .font(.body).scrollContentBackground(.hidden)
                        .frame(minHeight: 80)
                        .padding(6)
                        .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 6))
                        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color(nsColor: .separatorColor), lineWidth: 1))
                }

                OptimizeBar(engine: engine, prompt: $prompt, optimizationAttemptId: $optimizationAttemptId)

                referenceControl

                labeled("Resolution") {
                    Picker("", selection: $resolution) {
                        ForEach(VideoResolution.allCases) { Text($0.rawValue).tag($0) }
                    }.labelsHidden()
                }

                // Duration is the primary length control; frames are derived
                // and snapped to LTX's legal 1+8k counts. The read-out below
                // shows exactly what will be rendered so the snap is visible.
                NumericSliderField(label: "Duration (s)", value: $seconds, range: 1...12, step: 0.5, fractionDigits: 1)
                    .help("Clip length. Snapped to the nearest legal LTX frame count (1+8k, 25–289f). 289f (~12s) is the single-pass cap.")
                    .onChange(of: seconds) { _, s in
                        frames = Self.framesForSeconds(s)
                    }

                Text("→ \(frames) frames · \(String(format: "%.1fs", Self.secondsForFrames(frames))) at \(Int(Self.playbackFps))fps"
                     + (frames >= 289 ? "  (single-pass cap)" : ""))
                    .font(.caption)
                    .foregroundStyle(.secondary)

                // The ACTUAL motion dial. Distinct from playback fps: spreading
                // the same action over a lower cond_fps yields more movement.
                NumericSliderField(label: "Motion FPS (0 = auto)", value: $condFps, range: 0...30, step: 1)
                    .help("Temporal conditioning rate (cond_fps). Lower = more motion for the same action; higher = stiller. 0 leaves it to the engine (previous behaviour). Values below 6 are floored by the engine.")

                NumericSliderField(label: "Steps", value: $steps, range: 1...30, step: 1)
                if referencePath != nil {
                    NumericSliderField(label: "Strength", value: $strength, range: 0...1, step: 0.05, fractionDigits: 2)
                        .help("How hard the seed image is held. 0.5 = validated default. Higher loosens the pose lock but risks mid-clip drift/ghosting on long passes; lower holds the frame rigid.")
                    NumericSliderField(label: "Extend (s)", value: $extendSeconds, range: 0...12, step: 1)
                }

                labeled("Seed (empty = random)") {
                    TextField("Random", text: $seedText).textFieldStyle(.roundedBorder)
                }

                // Per-render overrides for knobs that otherwise live only in
                // the launchd plist as LTX2_* env vars (global + restart).
                // Everything here defaults to inherit, so leaving the group
                // untouched sends exactly what it sends today.
                DisclosureGroup("Advanced — per-render overrides", isExpanded: $showAdvanced) {
                    VStack(alignment: .leading, spacing: 10) {
                        labeled("Sampler (blank = inherit)") {
                            Picker("", selection: $advSampler) {
                                ForEach(Self.samplerOptions, id: \.self) { s in
                                    Text(s.isEmpty ? "inherit" : s).tag(s)
                                }
                            }.labelsHidden()
                        }
                        .help("LTX-2 sampler. Your engine default is LTX2_SAMPLER=euler_ancestral_cfg_pp (the PinkCherry-validated pairing).")

                        NumericSliderField(label: "Guidance rescale (-1 = inherit)", value: $advGuidanceRescale, range: -1...1, step: 0.05, fractionDigits: 2)
                        NumericSliderField(label: "Img compression, i2v (-1 = inherit)", value: $advImgCompression, range: -1...100, step: 1)
                            .help("LTX2_I2V_COMPRESSION. Lower preserves more of the seed frame's detail.")
                        NumericSliderField(label: "Color anchor (-1 = inherit)", value: $advColorAnchor, range: -1...1, step: 0.05, fractionDigits: 2)
                        NumericSliderField(label: "NAG scale (-1 = inherit)", value: $advNagScale, range: -1...50, step: 0.5, fractionDigits: 1)
                            .help("Normalized Attention Guidance — how PinkCherry gets prompt adherence at CFG 1.0.")
                        NumericSliderField(label: "NAG alpha (-1 = inherit)", value: $advNagAlpha, range: -1...1, step: 0.05, fractionDigits: 2)
                        NumericSliderField(label: "NAG tau (-1 = inherit)", value: $advNagTau, range: -1...10, step: 0.1, fractionDigits: 1)

                        labeled("Two-stage refine") {
                            Picker("", selection: $advTwoStage) {
                                Text("inherit").tag(-1)
                                Text("off").tag(0)
                                Text("on").tag(1)
                            }.pickerStyle(.segmented)
                        }
                        .help("Your engine default is LTX2_TWO_STAGE=0 (off) — refine renders SOFTER on this recipe.")

                        if !activeOverrideSummary.isEmpty {
                            Text("Overriding: \(activeOverrideSummary)")
                                .font(.caption).foregroundStyle(.orange)
                        }
                    }
                    .padding(.top, 6)
                }

                labeled("LoRAs") {
                    LoRAPicker(engine: engine, selectedLoras: $selectedLoras, familyOverride: "ltx")
                        .frame(minHeight: 180, maxHeight: 260)
                }

                VideoTuningPanel(tuning: $tuningOverrides)

                EffectiveConfigCard(engine: engine)

                PromptLabPanel(engine: engine)

                Button(action: generate) {
                    HStack {
                        if isGenerating { ProgressView().controlSize(.small).padding(.trailing, 2) }
                        Image(systemName: "film")
                        Text(isGenerating ? "Generating…" : "Generate Video")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(isGenerating || !engine.connectionState.isConnected
                          || prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                if let statusMessage {
                    Text(statusMessage).font(.caption).foregroundStyle(.secondary)
                }
                if let errorMessage {
                    Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption).foregroundStyle(.orange)
                }
                Text("Local video needs the server started with --ltx2-weights + --ltx2-gemma. Generation takes several minutes.")
                    .font(.caption2).foregroundStyle(.tertiary)
            }
            .padding(16)
        }
    }

    private var referenceControl: some View {
        labeled("Reference image (image-to-video, optional)") {
            if let path = referencePath {
                HStack(spacing: 10) {
                    Group {
                        if let thumb = referenceThumb {
                            Image(nsImage: thumb).resizable().aspectRatio(contentMode: .fill)
                        } else { Image(systemName: "photo").foregroundStyle(.tertiary) }
                    }
                    .frame(width: 64, height: 64).clipShape(RoundedRectangle(cornerRadius: 6))
                    Text((path as NSString).lastPathComponent).font(.caption).lineLimit(1).truncationMode(.middle)
                    Spacer()
                    Button("Remove", role: .destructive) { referencePath = nil; referenceThumb = nil }
                        .controlSize(.small)
                }
            } else {
                Button {
                    chooseReference()
                } label: { Label("Choose Image…", systemImage: "photo.badge.plus") }
                    .controlSize(.small)
                Text("Or drag a PNG in.")
                    .font(.caption2).foregroundStyle(.tertiary)
            }
        }
        .padding(6)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(isReferenceDropTargeted ? Color.accentColor : Color.clear, style: StrokeStyle(lineWidth: 2, dash: [5]))
        )
        .onDrop(of: [.fileURL, .image], isTargeted: $isReferenceDropTargeted) { providers in
            handleImageDrop(providers, apply: setReference)
        }
    }

    // MARK: - Preview

    private var preview: some View {
        ZStack {
            Color(nsColor: .controlBackgroundColor)
            if let url = resultURL, let player {
                VStack(spacing: 8) {
                    SafeVideoPlayer(player: player)
                        .frame(maxWidth: .infinity, minHeight: 300)
                    HStack {
                        Button { NSWorkspace.shared.selectFile(url.path, inFileViewerRootedAtPath: "") } label: {
                            Label("Reveal", systemImage: "magnifyingglass")
                        }
                        Button { NSWorkspace.shared.open(url) } label: {
                            Label("Open", systemImage: "play.rectangle")
                        }
                        Spacer()
                    }
                    .padding(.horizontal, 12).padding(.bottom, 8)
                }
            } else if isGenerating {
                VStack(spacing: 10) {
                    ProgressView().controlSize(.large)
                    Text(statusMessage ?? "Generating video…").foregroundStyle(.secondary)
                    Text("LTX-2 renders locally — this can take several minutes.")
                        .font(.caption).foregroundStyle(.tertiary)
                }
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "film").font(.system(size: 48)).foregroundStyle(.tertiary)
                    Text("Your video will appear here").foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: - Actions

    private func labeled<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            content()
        }
    }

    private func chooseReference() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true; panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.png, .jpeg, .image]
        panel.prompt = "Use as Reference"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        setReference(url.path)
    }

    private func setReference(_ path: String) {
        referencePath = path
        Task {
            let img = await Task.detached { NSImage(contentsOfFile: path) }.value
            await MainActor.run { referenceThumb = img }
        }
    }

    /// Prefill from Settings → Motion (once), matching a stored resolution to a
    /// preset when possible.
    private func applyDefaults() {
        guard !didApplyDefaults else { return }
        didApplyDefaults = true
        let s = DesktopSettings.load()
        if let f = s.videoFrames, Self.frameOptions.contains(f) { frames = f }
        // Keep the seconds control showing the same clip the saved frame count
        // represents, so the derived read-out is never out of step on launch.
        seconds = Self.secondsForFrames(frames)
        if let st = s.videoSteps { steps = Double(st) }
        if let w = s.videoWidth, let h = s.videoHeight,
           let match = VideoResolution.allCases.first(where: { $0.size == (w, h) }) {
            resolution = match
        }
    }

    private func consumePendingReference() {
        guard let path = pendingMotionReference, !path.isEmpty else { return }
        pendingMotionReference = nil
        setReference(path)
    }

    private func generate() {
        let (w, h) = resolution.size
        let seed = UInt64(seedText) ?? 0
        let outputDir = DesktopSettings.load().outputDirectory
        let outputPath = (outputDir as NSString).appendingPathComponent("ltx2-\(Int(Date().timeIntervalSince1970)).mp4")

        isGenerating = true
        errorMessage = nil
        player?.pause()
        player = nil
        resultURL = nil
        statusMessage = "Loading LTX-2 and generating \(frames) frames…"

        // Merge the UI's motion dial into any tuning already staged (e.g. from
        // a preset). An explicit control wins; 0 means "auto" so the key is
        // omitted entirely and the engine keeps choosing, exactly as before.
        var tuning = tuningOverrides
        if condFps > 0 { tuning["cond_fps"] = condFps }
        // Advanced group wins over staged tuning — it is the most explicit
        // expression of intent for THIS render.
        for (k, v) in advancedOverrides { tuning[k] = v }

        let request = EngineService.VideoRequest(
            prompt: prompt,
            initImagePath: referencePath,
            width: w, height: h, frames: frames,
            steps: Int(steps), seed: seed, strength: Float(strength),
            extendToSeconds: Float(extendSeconds),
            loras: selectedLoras,
            outputPath: outputPath,
            tuning: tuning.isEmpty ? nil : tuning,
            optimizationAttemptId: optimizationAttemptId
        )

        Task {
            do {
                // Submit + poll: a multi-minute / multi-chunk render never holds
                // the HTTP connection open, so the UI stays live and shows real
                // per-chunk/per-step progress instead of freezing on one request.
                let jobId = try await engine.submitVideoJob(request)
                statusMessage = "Queued LTX-2 job — loading model…"
                let result = try await engine.pollVideoStatus(jobId: jobId) { job in
                    if let pct = job.progressPercent, job.status == "processing" {
                        statusMessage = "Rendering \(frames) frames… \(pct)%"
                    } else if job.status == "processing" {
                        statusMessage = "Rendering \(frames) frames…"
                    }
                }
                statusMessage = String(format: "Done — %d frames, %.1fs video in %.0fs",
                                       result.frameCount, result.durationSeconds, result.elapsedSeconds)
                let url = URL(fileURLWithPath: result.outputPath)
                player = AVPlayer(url: url)
                resultURL = url
            } catch {
                errorMessage = error.localizedDescription
                statusMessage = nil
            }
            isGenerating = false
        }
    }
}
