// AgentView.swift — Chat assistant panel for image creation
//
// A conversational helper (Dan's v1.3 via LM Studio) that refines prompts and
// advises on framing/lighting/params. Assistant turns that carry a "PROMPT:"
// suggestion get a "Use in Generate" action.

import SwiftUI

struct AgentView: View {
    @Bindable var agent: AgentService
    /// Push a prompt to the Generate tab.
    var onUsePrompt: ((String) -> Void)?

    @State private var input: String = ""
    @FocusState private var inputFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            transcript
            Divider()
            composer
        }
        .navigationTitle("Assistant")
        .onAppear { inputFocused = true }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "sparkles").foregroundStyle(.purple)
            Text("Image Assistant").font(.headline)
            if let model = agent.modelName {
                Text(model).font(.caption2).foregroundStyle(.tertiary).lineLimit(1)
            }
            Spacer()
            Button { agent.reset() } label: { Label("New Chat", systemImage: "square.and.pencil") }
                .buttonStyle(.borderless)
                .disabled(agent.messages.isEmpty)
        }
        .padding(12)
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    if agent.messages.isEmpty {
                        emptyState
                    }
                    ForEach(agent.messages) { message in
                        messageRow(message).id(message.id)
                    }
                    if agent.isThinking {
                        HStack(spacing: 6) {
                            ProgressView().controlSize(.small)
                            Text("Thinking…").font(.caption).foregroundStyle(.secondary)
                        }
                        .id("thinking")
                    }
                    if let error = agent.lastError {
                        Label(error, systemImage: "exclamationmark.triangle.fill")
                            .font(.caption).foregroundStyle(.orange)
                    }
                }
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .onChange(of: agent.messages.count) { _, _ in
                withAnimation { proxy.scrollTo(agent.messages.last?.id, anchor: .bottom) }
            }
            .onChange(of: agent.isThinking) { _, thinking in
                if thinking { withAnimation { proxy.scrollTo("thinking", anchor: .bottom) } }
            }
        }
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Ask for help crafting an image.").font(.headline).foregroundStyle(.secondary)
            ForEach([
                "A moody portrait of Kira in the coffee shop at golden hour",
                "Give me 3 prompt variations for a cyberpunk street scene",
                "What camera and lighting for a dramatic low-angle hero shot?",
            ], id: \.self) { suggestion in
                Button {
                    Task { await agent.send(suggestion) }
                } label: {
                    Text(suggestion).font(.callout).multilineTextAlignment(.leading)
                }
                .buttonStyle(.plain)
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
            }
        }
        .padding(.vertical, 20)
    }

    @ViewBuilder
    private func messageRow(_ message: AgentMessage) -> some View {
        let isUser = message.role == .user
        HStack {
            if isUser { Spacer(minLength: 40) }
            VStack(alignment: .leading, spacing: 6) {
                Text(message.text)
                    .font(.callout)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                if !isUser {
                    // Prefer a marked PROMPT: line; otherwise offer the whole
                    // reply (Dan's enhancer usually emits a prompt directly).
                    let prompt = AgentService.suggestedPrompt(from: message.text) ?? message.text
                    Button {
                        onUsePrompt?(prompt)
                    } label: {
                        Label("Use in Generate", systemImage: "wand.and.stars")
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                }
            }
            .padding(10)
            .background(
                isUser ? Color.accentColor.opacity(0.15) : Color(nsColor: .controlBackgroundColor),
                in: RoundedRectangle(cornerRadius: 10)
            )
            if !isUser { Spacer(minLength: 40) }
        }
    }

    private var composer: some View {
        HStack(spacing: 8) {
            TextField("Ask the assistant…", text: $input, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...5)
                .focused($inputFocused)
                .onSubmit(send)
                .padding(8)
                .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(nsColor: .separatorColor), lineWidth: 1))
            Button(action: send) {
                Image(systemName: "arrow.up.circle.fill").font(.title2)
            }
            .buttonStyle(.plain)
            .disabled(input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || agent.isThinking)
        }
        .padding(12)
    }

    private func send() {
        let text = input
        input = ""
        Task { await agent.send(text) }
    }
}

// MARK: - Compact panel embedded in Generate

/// A small assistant chat that lives inside the Generate control panel. When
/// a reply carries a parameter action, `onApply` is invoked so the assistant
/// can populate the generation controls (and optionally start a render).
struct GenerateAssistantPanel: View {
    @Bindable var agent: AgentService
    var onApply: (AgentAction) -> Void

    @State private var input: String = ""
    /// Ids of assistant messages whose action was already applied.
    @State private var appliedMessageIds: Set<String> = []

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if agent.messages.isEmpty {
                Text("Ask the assistant to set up a shot, e.g. \"portrait of Kira, 85mm, 1024×1536, 9 steps\" — it fills the controls below.")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 8) {
                            ForEach(agent.messages) { message in
                                messageRow(message).id(message.id)
                            }
                        }
                        .padding(.vertical, 2)
                    }
                    .frame(maxHeight: 240)
                    .onChange(of: agent.messages.count) { _, _ in
                        withAnimation { proxy.scrollTo(agent.messages.last?.id, anchor: .bottom) }
                    }
                }
            }

            if agent.isThinking {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text("Thinking…").font(.caption).foregroundStyle(.secondary)
                }
            }
            if let error = agent.lastError {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption2).foregroundStyle(.orange).lineLimit(2)
            }

            HStack(spacing: 6) {
                TextField("Ask the assistant…", text: $input, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(1...4)
                    .onSubmit(send)
                Button(action: send) { Image(systemName: "arrow.up.circle.fill").font(.title3) }
                    .buttonStyle(.plain)
                    .disabled(input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || agent.isThinking)
                if !agent.messages.isEmpty {
                    Button { agent.reset() } label: { Image(systemName: "square.and.pencil") }
                        .buttonStyle(.plain)
                        .help("New chat")
                }
            }
        }
        // Apply an action as soon as a new assistant reply carries one.
        .onChange(of: agent.messages.count) { _, _ in applyLatestActionIfNeeded() }
    }

    @ViewBuilder
    private func messageRow(_ message: AgentMessage) -> some View {
        let isUser = message.role == .user
        VStack(alignment: .leading, spacing: 4) {
            Text(message.text)
                .font(.caption)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            if !isUser, let action = AgentService.parseAction(from: message.text) {
                Button {
                    onApply(action)
                    appliedMessageIds.insert(message.id)
                } label: {
                    Label("Apply: \(action.summary)", systemImage: "slider.horizontal.3")
                        .font(.caption2)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.mini)
            }
        }
        .padding(8)
        .background(
            isUser ? Color.accentColor.opacity(0.12) : Color(nsColor: .controlBackgroundColor),
            in: RoundedRectangle(cornerRadius: 8)
        )
    }

    private func send() {
        let text = input
        input = ""
        Task { await agent.send(text) }
    }

    /// When the newest message is an assistant reply with an unapplied action,
    /// apply it automatically (the button remains for re-applying).
    private func applyLatestActionIfNeeded() {
        guard let last = agent.messages.last, last.role == .assistant,
              !appliedMessageIds.contains(last.id),
              let action = AgentService.parseAction(from: last.text)
        else { return }
        appliedMessageIds.insert(last.id)
        onApply(action)
    }
}
