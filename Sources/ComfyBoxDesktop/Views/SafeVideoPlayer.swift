// SafeVideoPlayer.swift — AppKit-backed video player, bypassing _AVKit_SwiftUI.
//
// SwiftUI's `VideoPlayer` bridges through the private `_AVKit_SwiftUI`
// framework. On the macOS 27 (Tahoe) beta that framework has a broken class
// hierarchy: the Swift runtime aborts during generic metadata init when
// SwiftUI's layout pipeline instantiates `VideoPlayer` (issue #257 — 29
// crashes, no ComfyBox code on the stack, all frames inside
// libswiftCore/SwiftUI/AttributeGraph/_AVKit_SwiftUI).
//
// This wraps AVKit's AppKit `AVPlayerView` directly via NSViewRepresentable,
// which never touches `_AVKit_SwiftUI`. It's a permanent replacement, not a
// beta-only guard — safe on every macOS version, so there's nothing to
// revisit once Apple ships a fix.
//
// Same contract as SwiftUI's `VideoPlayer(player:)`: pass a pre-owned
// `AVPlayer?` built once (e.g. when a result URL lands), never construct one
// inline in the view body — doing that still churns a new player on every
// re-render regardless of which player view renders it.

import AVKit
import SwiftUI

struct SafeVideoPlayer: NSViewRepresentable {
    let player: AVPlayer?
    var controlsStyle: AVPlayerViewControlsStyle = .inline

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.controlsStyle = controlsStyle
        view.player = player
        return view
    }

    func updateNSView(_ nsView: AVPlayerView, context: Context) {
        if nsView.player !== player {
            nsView.player = player
        }
    }
}
