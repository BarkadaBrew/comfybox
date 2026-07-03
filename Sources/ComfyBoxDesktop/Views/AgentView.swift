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
