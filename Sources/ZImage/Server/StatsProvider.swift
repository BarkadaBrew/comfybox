// StatsProvider.swift — Server stats, process/system memory, and a provider-config
// summary for the ComfyBox warm server.
//
// Swift port of the retiring Coffee Shop image service's `src/memory-guard.ts`
// (memory pressure thresholds + RSS/free/total sampling) and `src/providers-status.ts`
// (per-provider health/configuration report). This file owns only the *pure* data
// gathering and formatting — no HTTP route wiring, no network I/O. A later Routes
// agent surfaces `ServerStatsSnapshot` on an endpoint.
//
// The pure/live split keeps threshold and formatting logic unit-testable without
// asserting on live machine memory:
//   * `MemoryPressureThresholds.evaluate(...)` — pure classification
//   * `StatsProvider.memoryStatus(processRssBytes:systemFreeBytes:systemTotalBytes:)` — pure conversion
//   * `StatsProvider.snapshot(memory:uptimeSeconds:...)` — pure assembly
//   * `MemoryProbe.*` / `StatsProvider.sampleMemoryStatus()` / `liveSnapshot(...)` — live sampling

import Foundation

// MARK: - Memory pressure

/// Memory-pressure classification. Mirrors `MemoryStatus['pressureLevel']` in
/// `memory-guard.ts` (`'normal' | 'warning' | 'critical'`).
public enum MemoryPressureLevel: String, Codable, Sendable {
  case normal
  case warning
  case critical
}

/// RSS / free-memory thresholds that classify memory pressure. Defaults mirror
/// `DEFAULT_CONFIG` in `memory-guard.ts` (8 GB warn, 12 GB / 2 GB free critical).
public struct MemoryPressureThresholds: Codable, Equatable, Sendable {
  /// Process RSS (MB) at or above which pressure becomes `.warning`. Default 8192 (8 GB).
  public var warningRssMb: Int
  /// Process RSS (MB) at or above which pressure becomes `.critical`. Default 12288 (12 GB).
  public var criticalRssMb: Int
  /// System available memory (MB) at or below which pressure becomes `.critical`. Default 2048 (2 GB).
  public var criticalFreeMb: Int

  public init(warningRssMb: Int = 8192, criticalRssMb: Int = 12288, criticalFreeMb: Int = 2048) {
    self.warningRssMb = warningRssMb
    self.criticalRssMb = criticalRssMb
    self.criticalFreeMb = criticalFreeMb
  }

  /// Default thresholds matching the Node image service.
  public static let `default` = MemoryPressureThresholds()

  /// Classify pressure from a process RSS and system-free reading (both in MB).
  ///
  /// Ordering matches `MemoryGuard.getStatus()`: critical wins over warning, and a
  /// low free-memory reading forces critical even when RSS is modest.
  public func evaluate(processRssMb: Int, systemFreeMb: Int) -> MemoryPressureLevel {
    if processRssMb >= criticalRssMb || systemFreeMb <= criticalFreeMb {
      return .critical
    }
    if processRssMb >= warningRssMb {
      return .warning
    }
    return .normal
  }
}

/// Snapshot of process + system memory. Mirrors `MemoryStatus` in `memory-guard.ts`.
public struct MemoryStatus: Codable, Equatable, Sendable {
  /// Resident process footprint in MB (`task_vm_info.phys_footprint`).
  public var processRssMb: Int
  /// System available memory in MB (free + inactive + speculative + purgeable pages).
  public var systemFreeMb: Int
  /// Total physical memory in MB.
  public var systemTotalMb: Int
  /// Derived pressure classification.
  public var pressureLevel: MemoryPressureLevel

  public init(processRssMb: Int, systemFreeMb: Int, systemTotalMb: Int, pressureLevel: MemoryPressureLevel) {
    self.processRssMb = processRssMb
    self.systemFreeMb = systemFreeMb
    self.systemTotalMb = systemTotalMb
    self.pressureLevel = pressureLevel
  }
}

// MARK: - Providers summary

/// Configuration state of one AI capability. This is a *config* summary — it reports
/// whether a capability is wired up, not whether it is reachable right now (live
/// reachability probes require async network I/O and belong to the Routes layer).
public struct ProviderCapabilityStatus: Codable, Equatable, Sendable {
  /// True when the capability has a usable endpoint / credential in config.
  public var configured: Bool
  /// Human-readable hint (endpoint model/url, or why it's unconfigured).
  public var detail: String?

  public init(configured: Bool, detail: String? = nil) {
    self.configured = configured
    self.detail = detail
  }
}

/// Which ComfyBox capabilities are configured. Mirrors the per-provider shape of
/// `ProviderStatusReport` in `providers-status.ts`, restricted to what is knowable
/// from config alone.
public struct ProvidersStatusSummary: Codable, Equatable, Sendable {
  public var promptOptimization: ProviderCapabilityStatus
  public var vision: ProviderCapabilityStatus
  public var captioning: ProviderCapabilityStatus
  public var replicate: ProviderCapabilityStatus

  public init(
    promptOptimization: ProviderCapabilityStatus,
    vision: ProviderCapabilityStatus,
    captioning: ProviderCapabilityStatus,
    replicate: ProviderCapabilityStatus
  ) {
    self.promptOptimization = promptOptimization
    self.vision = vision
    self.captioning = captioning
    self.replicate = replicate
  }

  /// Number of capabilities that are configured.
  public var configuredCount: Int {
    [promptOptimization, vision, captioning, replicate].reduce(0) { $0 + ($1.configured ? 1 : 0) }
  }
}

/// Summarizes which capabilities in a ``ComfyBoxServerConfig`` are configured.
public enum ProvidersStatusSummarizer {
  /// Build a config-only provider summary.
  public static func summarize(_ config: ComfyBoxServerConfig) -> ProvidersStatusSummary {
    ProvidersStatusSummary(
      promptOptimization: endpointStatus(config.providers.promptOptimization),
      vision: endpointStatus(config.providers.vision),
      captioning: endpointStatus(config.providers.captioning),
      replicate: replicateStatus(config.replicate)
    )
  }

  /// An OpenAI-compatible capability is configured when it has a non-empty base URL.
  static func endpointStatus(_ endpoint: AIProviderEndpoint?) -> ProviderCapabilityStatus {
    guard let endpoint,
          !endpoint.baseUrl.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      return ProviderCapabilityStatus(configured: false, detail: nil)
    }
    let model = endpoint.model.trimmingCharacters(in: .whitespacesAndNewlines)
    let modelPart = model.isEmpty ? "?" : model
    return ProviderCapabilityStatus(
      configured: true,
      detail: "model=\(modelPart) baseUrl=\(endpoint.baseUrl)"
    )
  }

  /// Replicate is configured when an API key is present (mirrors the `not_configured`
  /// branch of `checkReplicate` in `providers-status.ts`).
  static func replicateStatus(_ replicate: ReplicateProviderConfig?) -> ProviderCapabilityStatus {
    guard let replicate,
          let apiKey = replicate.apiKey,
          !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      return ProviderCapabilityStatus(
        configured: false,
        detail: "replicate.apiKey is empty — set it in ~/.comfybox/config.json"
      )
    }
    var bits = ["apiKey set"]
    if let imageModel = replicate.imageModel, !imageModel.isEmpty { bits.append("imageModel=\(imageModel)") }
    if let videoModel = replicate.videoModel, !videoModel.isEmpty { bits.append("videoModel=\(videoModel)") }
    return ProviderCapabilityStatus(configured: true, detail: bits.joined(separator: " "))
  }
}

// MARK: - Combined snapshot

/// A complete stats snapshot for the server. Codable so a route can return it directly.
public struct ServerStatsSnapshot: Codable, Equatable, Sendable {
  public var uptimeSeconds: Int
  /// Successful render count, when the caller tracks it.
  public var renderCount: Int?
  /// Failed render count, when the caller tracks it.
  public var failedRenderCount: Int?
  /// Queued/pending request count, when the caller tracks it.
  public var pendingCount: Int?
  public var memory: MemoryStatus
  public var providers: ProvidersStatusSummary

  public init(
    uptimeSeconds: Int,
    renderCount: Int? = nil,
    failedRenderCount: Int? = nil,
    pendingCount: Int? = nil,
    memory: MemoryStatus,
    providers: ProvidersStatusSummary
  ) {
    self.uptimeSeconds = uptimeSeconds
    self.renderCount = renderCount
    self.failedRenderCount = failedRenderCount
    self.pendingCount = pendingCount
    self.memory = memory
    self.providers = providers
  }
}

// MARK: - Provider

/// Gathers server stats: process/system memory, uptime, render counts, and a
/// provider-config summary. Pure builders are separated from live samplers so the
/// classification and formatting logic can be unit-tested deterministically.
public struct StatsProvider: Sendable {
  public var thresholds: MemoryPressureThresholds

  public init(thresholds: MemoryPressureThresholds = .default) {
    self.thresholds = thresholds
  }

  // MARK: Pure builders

  /// Convert raw byte readings into a classified ``MemoryStatus``. Pure — no sampling.
  public func memoryStatus(
    processRssBytes: UInt64,
    systemFreeBytes: UInt64,
    systemTotalBytes: UInt64
  ) -> MemoryStatus {
    let bytesPerMb: UInt64 = 1024 * 1024
    let processRssMb = Int(processRssBytes / bytesPerMb)
    let systemFreeMb = Int(systemFreeBytes / bytesPerMb)
    let systemTotalMb = Int(systemTotalBytes / bytesPerMb)
    return MemoryStatus(
      processRssMb: processRssMb,
      systemFreeMb: systemFreeMb,
      systemTotalMb: systemTotalMb,
      pressureLevel: thresholds.evaluate(processRssMb: processRssMb, systemFreeMb: systemFreeMb)
    )
  }

  /// Assemble a full snapshot from an already-sampled ``MemoryStatus``. Pure — no sampling.
  public func snapshot(
    memory: MemoryStatus,
    uptimeSeconds: Int,
    renderCount: Int? = nil,
    failedRenderCount: Int? = nil,
    pendingCount: Int? = nil,
    config: ComfyBoxServerConfig
  ) -> ServerStatsSnapshot {
    ServerStatsSnapshot(
      uptimeSeconds: max(0, uptimeSeconds),
      renderCount: renderCount,
      failedRenderCount: failedRenderCount,
      pendingCount: pendingCount,
      memory: memory,
      providers: ProvidersStatusSummarizer.summarize(config)
    )
  }

  /// Whole-number uptime seconds from a start time. Pure.
  public static func uptimeSeconds(startTime: Date, now: Date = Date()) -> Int {
    max(0, Int(now.timeIntervalSince(startTime)))
  }

  // MARK: Live samplers

  /// Sample process + system memory right now and classify it.
  public func sampleMemoryStatus() -> MemoryStatus {
    memoryStatus(
      processRssBytes: MemoryProbe.processFootprintBytes(),
      systemFreeBytes: MemoryProbe.systemAvailableMemoryBytes(),
      systemTotalBytes: MemoryProbe.systemTotalMemoryBytes()
    )
  }

  /// Sample everything live and assemble a snapshot.
  public func liveSnapshot(
    startTime: Date,
    now: Date = Date(),
    renderCount: Int? = nil,
    failedRenderCount: Int? = nil,
    pendingCount: Int? = nil,
    config: ComfyBoxServerConfig
  ) -> ServerStatsSnapshot {
    snapshot(
      memory: sampleMemoryStatus(),
      uptimeSeconds: StatsProvider.uptimeSeconds(startTime: startTime, now: now),
      renderCount: renderCount,
      failedRenderCount: failedRenderCount,
      pendingCount: pendingCount,
      config: config
    )
  }
}

// MARK: - Low-level memory probes

/// Mach-level memory readings. Kept separate from the pure logic so tests never
/// depend on live machine state.
public enum MemoryProbe {
  /// Resident process footprint in bytes (`task_vm_info.phys_footprint`), matching
  /// the probe WarmServer already uses. Returns 0 on failure.
  public static func processFootprintBytes() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout.size(ofValue: info) / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { pointer in
      pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), rebound, &count)
      }
    }
    guard result == KERN_SUCCESS else { return 0 }
    return info.phys_footprint
  }

  /// System *available* memory in bytes: free + inactive + purgeable pages. Mirrors
  /// `getAvailableMemoryBytes()` in `memory-guard.ts` (which parses `vm_stat` and sums
  /// free + inactive + speculative + purgeable) but reads
  /// `host_statistics64(HOST_VM_INFO64)` directly. NOTE: unlike the `vm_stat` CLI,
  /// `host_statistics64`'s `free_count` already includes speculative pages (see
  /// `vm_statistics.h`), so speculative is intentionally not added again here — that
  /// keeps this total equal to the Node figure without double-counting. On macOS a
  /// "free only" reading wildly under-reports with a warm ~20 GB model resident,
  /// because the kernel keeps everything else as reclaimable inactive cache.
  /// Returns 0 on failure.
  public static func systemAvailableMemoryBytes() -> UInt64 {
    var stats = vm_statistics64_data_t()
    var count = mach_msg_type_number_t(
      MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size
    )
    let hostPort = mach_host_self()
    let result = withUnsafeMutablePointer(to: &stats) { pointer in
      pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
        host_statistics64(hostPort, HOST_VM_INFO64, rebound, &count)
      }
    }
    guard result == KERN_SUCCESS else { return 0 }
    let pageSize = UInt64(vm_kernel_page_size)
    let available =
      UInt64(stats.free_count)
      + UInt64(stats.inactive_count)
      + UInt64(stats.purgeable_count)
    return available * pageSize
  }

  /// Total physical memory in bytes.
  public static func systemTotalMemoryBytes() -> UInt64 {
    ProcessInfo.processInfo.physicalMemory
  }
}
