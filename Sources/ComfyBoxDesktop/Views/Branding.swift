// Branding.swift — CoffeeShop Desktop logo, splash, and About
//
// The app's own icon doubles as the logo so branding stays consistent
// everywhere (Dock, splash, About). Falls back to an SF Symbol during dev when
// no bundle icon is set.

import SwiftUI
import AppKit

enum Branding {
    static let appName = "CoffeeShop Desktop"
    static let tagline = "The creative studio for the Coffeeshop suite"

    static var version: String {
        let v = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String
        return v ?? "1.0"
    }

    /// The app icon as a logo image (nil during dev if unset).
    static var logo: NSImage? {
        let img = NSApplication.shared.applicationIconImage
        // The generic placeholder icon is usually tiny/absent; treat missing.
        return (img?.size.width ?? 0) > 0 ? img : nil
    }

    @ViewBuilder
    static func logoView(size: CGFloat) -> some View {
        if let logo {
            Image(nsImage: logo).resizable().frame(width: size, height: size)
        } else {
            Image(systemName: "cup.and.saucer.fill")
                .font(.system(size: size * 0.6))
                .frame(width: size, height: size)
                .foregroundStyle(.brown)
        }
    }
}

/// Brief launch splash; fades out on its own.
struct SplashView: View {
    @Binding var isPresented: Bool
    @State private var opacity: Double = 0

    var body: some View {
        ZStack {
            LinearGradient(colors: [Color(red: 1.0, green: 0.85, blue: 0.78),
                                    Color(red: 0.80, green: 0.90, blue: 0.95),
                                    Color(red: 0.85, green: 0.80, blue: 0.95)],
                           startPoint: .topLeading, endPoint: .bottomTrailing)
                .ignoresSafeArea()
            VStack(spacing: 16) {
                Branding.logoView(size: 132)
                    .clipShape(RoundedRectangle(cornerRadius: 28))
                    .shadow(radius: 12, y: 6)
                Text(Branding.appName).font(.system(size: 28, weight: .semibold))
                Text(Branding.tagline).font(.callout).foregroundStyle(.secondary)
            }
            .opacity(opacity)
        }
        .onAppear {
            withAnimation(.easeOut(duration: 0.4)) { opacity = 1 }
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                withAnimation(.easeIn(duration: 0.4)) { opacity = 0 }
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.45) { isPresented = false }
            }
        }
    }
}

/// About window content.
struct AboutView: View {
    var body: some View {
        VStack(spacing: 14) {
            Branding.logoView(size: 96)
                .clipShape(RoundedRectangle(cornerRadius: 20))
                .shadow(radius: 8, y: 4)
            VStack(spacing: 3) {
                Text(Branding.appName).font(.title2.weight(.semibold))
                Text("Version \(Branding.version)").font(.caption).foregroundStyle(.secondary)
            }
            Text("The hub for the entire Coffeeshop suite — generate images, video, and voice; manage models & LoRAs; run mflux, Découpage, and face identity; monitor the stack; and work with Bree.")
                .font(.callout).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: 360)
            Text("Powered by the ComfyBox engine (Z-Image / MLX on Apple Silicon).")
                .font(.caption2).foregroundStyle(.tertiary)
        }
        .padding(28)
        .frame(width: 420)
    }
}
