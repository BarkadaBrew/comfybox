// BreeView.swift — Bree companion cockpit (handoff channel)
//
// Two panes: Bree's replies (bree-to-desktop.md) on the left, and a composer +
// history for messages to Bree (desktop-to-bree.md) on the right.

import SwiftUI
import AppKit

struct BreeView: View {
    @Bindable var bree: BreeService

    @State private var draft: String = ""

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if !bree.isAvailable {
                notAvailable
            } else {
                HSplitView {
                    inboxPane.frame(minWidth: 320)
                    composePane.frame(minWidth: 320)
                }
            }
        }
        .navigationTitle("Bree")
        .onAppear { bree.reload() }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "brain.head.profile").foregroundStyle(.pink)
            Text("Bree").font(.headline)
            Text("handoff channel").font(.caption).foregroundStyle(.secondary)
            Spacer()
            if let ts = bree.lastLoaded {
                Text("loaded \(ts.formatted(date: .omitted, time: .shortened))")
                    .font(.caption2).foregroundStyle(.tertiary)
            }
            Button { bree.reload() } label: { Image(systemName: "arrow.clockwise") }
                .help("Reload from vault")
            Button { NSWorkspace.shared.selectFile(bree.outboxPath, inFileViewerRootedAtPath: "") } label: {
                Image(systemName: "folder")
            }.help("Reveal handoff files")
        }
        .padding(12)
    }

    private var notAvailable: some View {
        VStack(spacing: 10) {
            Image(systemName: "questionmark.folder").font(.largeTitle).foregroundStyle(.orange)
            Text("Handoff folder not found").font(.headline)
            Text(bree.handoffDirectory).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity).padding()
    }

    private var inboxPane: some View {
        VStack(alignment: .leading, spacing: 0) {
            paneTitle("From Bree", systemImage: "arrow.down.left")
            ScrollView {
                Text(bree.inbox.isEmpty ? "No messages from Bree yet." : bree.inbox)
                    .font(.system(.callout, design: bree.inbox.isEmpty ? .default : .monospaced))
                    .foregroundStyle(bree.inbox.isEmpty ? .secondary : .primary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(12)
            }
        }
    }

    private var composePane: some View {
        VStack(alignment: .leading, spacing: 8) {
            paneTitle("To Bree", systemImage: "arrow.up.right")
            TextEditor(text: $draft)
                .font(.callout)
                .frame(minHeight: 100)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color(nsColor: .separatorColor)))
                .padding(.horizontal, 12)
            HStack {
                if let err = bree.lastError {
                    Label(err, systemImage: "exclamationmark.triangle").font(.caption2).foregroundStyle(.orange)
                }
                Spacer()
                Button("Send to Bree") { send() }
                    .buttonStyle(.borderedProminent)
                    .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            .padding(.horizontal, 12)

            Divider().padding(.top, 4)
            Text("History").font(.caption.weight(.semibold)).foregroundStyle(.secondary).padding(.horizontal, 12)
            ScrollView {
                Text(bree.outbox.isEmpty ? "Nothing sent yet." : bree.outbox)
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(bree.outbox.isEmpty ? .secondary : .primary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(12)
            }
        }
    }

    private func paneTitle(_ title: String, systemImage: String) -> some View {
        HStack(spacing: 6) {
            Image(systemName: systemImage).font(.caption).foregroundStyle(.secondary)
            Text(title).font(.subheadline.weight(.semibold))
        }
        .padding(.horizontal, 12).padding(.top, 8)
    }

    private func send() {
        bree.send(draft)
        draft = ""
    }
}
