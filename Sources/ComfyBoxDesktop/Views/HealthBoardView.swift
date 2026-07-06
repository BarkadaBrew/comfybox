// HealthBoardView.swift — "Datadog, but local" health board
//
// One pane answering "is the stack healthy right now?" without SSH
// (coffeeshop-dashboard-prd). Shows the ComfyBox engine, user-watched
// service endpoints, host unified memory, and a state-transition event
// timeline fed by HealthMonitor. Status is always icon + label + color,
// never color alone.

import SwiftUI
import Charts

/// Trailing window for activity stats and uptime figures.
enum HealthRange: String, CaseIterable, Identifiable {
    case week = "7d"
    case month = "30d"
    case all = "All"
    var id: String { rawValue }

    /// Days in the window; `nil` = unbounded (all history).
    var days: Int? {
        switch self {
        case .week: return 7
        case .month: return 30
        case .all: return nil
        }
    }
}

struct HealthBoardView: View {
    @Bindable var engine: EngineService
    @Bindable var monitor: HealthMonitor
    var store: DAMStore?

    @State private var newServiceName: String = ""
    @State private var newServiceURL: String = ""
    @State private var showAddService: Bool = false
    // Service control
    private let controller = ServiceController()
    @State private var busyServiceId: String?
    @State private var controlToast: (service: String, message: String, isError: Bool)?
    @State private var editingControl: WatchedService?

    // Render-activity data (from the DAM).
    @State private var renderTimestamps: [Date] = []
    @State private var range: HealthRange = .month

    // Littleroundbox server health (get_server_health consumer).
    @State private var serverHealth = ServerHealthService()

    // littleroundbox server metrics (Netdata).
    @State private var serverMetrics: ServerMetrics?
    @State private var serverMetricsFetchedAt: Date?
    private static let netdataURL = URL(string: "http://10.0.100.232:19999")!

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                header
                comfyBoxSection
                littleroundboxSection
                serverHealthSection
                trendsSection
                activitySection
                servicesSection
                memorySection
                eventsSection
            }
            .padding(20)
        }
        .navigationTitle("Health")
        .task { await loadActivity() }
        .task { await pollServerMetrics() }
        .task {
            if let ep = DesktopSettings.load().serverHealthEndpoint, !ep.isEmpty {
                serverHealth.endpoint = ep
            }
            serverHealth.startPolling(every: 60)
        }
    }

    /// Refresh littleroundbox metrics every 15s while the pane is visible.
    private func pollServerMetrics() async {
        let client = NetdataClient(baseURL: Self.netdataURL)
        while !Task.isCancelled {
            if let metrics = await client.fetchMetrics() {
                serverMetrics = metrics
                serverMetricsFetchedAt = Date()
            } else {
                serverMetrics = nil
            }
            try? await Task.sleep(for: .seconds(15))
        }
    }

    private func loadActivity() async {
        guard let store else { return }
        renderTimestamps = (try? await store.assetCreationTimestamps()) ?? []
    }

    // MARK: - Header

    /// Overall stack state: worst of the watched services and the engine.
    private var stackState: HealthState {
        let engineState = engineHealthState
        let monitorState = monitor.overallState
        return engineState.severityRank >= monitorState.severityRank ? engineState : monitorState
    }

    private var header: some View {
        HStack(spacing: 12) {
            statusBadge(stackState, font: .title3)
            if let refreshed = monitor.lastRefresh {
                Text("Checked \(refreshed, style: .relative) ago")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button {
                Task { await monitor.checkNow() }
            } label: {
                Label("Check Now", systemImage: "arrow.clockwise")
            }
        }
    }

    // MARK: - ComfyBox engine

    /// Engine health from live connection + queue state (no HTTP probe needed).
    private var engineHealthState: HealthState {
        guard engine.connectionState.isConnected else { return .down }
        if engine.queueInfo?.lastError != nil { return .degraded }
        return .healthy
    }

    private var comfyBoxSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionTitle("ComfyBox")
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 12)], spacing: 12) {
                let q = engine.queueInfo
                serviceCard(
                    service: WatchedService(
                        id: "comfybox-engine",
                        name: "ComfyBox Server",
                        urlString: "\(engine.serverHost):\(engine.serverPort)",
                        control: ServiceControl(launchdLabel: "com.barkadabrew.comfybox")),
                    state: engineHealthState,
                    detail: engine.connectionState.isConnected
                        ? "\(engine.serverHost):\(engine.serverPort)"
                        : "disconnected",
                    latencyMs: nil
                )
                StatTile(title: "Queue", value: "\(q?.pendingCount ?? 0)", systemImage: "tray.full")
                StatTile(
                    title: "Status",
                    value: (q?.isRendering ?? false) ? "Rendering" : "Idle",
                    systemImage: "bolt.fill",
                    tint: (q?.isRendering ?? false) ? .orange : .green
                )
                if let ms = q?.lastRenderDurationMs {
                    StatTile(title: "Last render", value: String(format: "%.1fs", Double(ms) / 1000), systemImage: "timer")
                }
                if let mem = q?.memoryUsageMB, mem > 0 {
                    StatTile(title: "Server memory", value: String(format: "%.1f GB", Double(mem) / 1024), systemImage: "memorychip")
                }
            }
            if let q = engine.queueInfo, q.isRendering, let pct = q.progressPercent {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(q.currentJobId.map { "Job \($0)" } ?? "Current render")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text("\(Int(pct))%")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                    ProgressView(value: min(max(pct / 100, 0), 1))
                }
            }
            if let error = engine.queueInfo?.lastError {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .lineLimit(2)
            }
        }
    }

    // MARK: - littleroundbox server (Netdata)

    private var littleroundboxSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                sectionTitle("littleroundbox")
                if let metrics = serverMetrics {
                    if let uptime = metrics.uptimeSeconds {
                        Text("up \(uptimeLabel(uptime))")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } else {
                    Label("Unreachable", systemImage: "wifi.slash")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
                Spacer()
                if let fetched = serverMetricsFetchedAt {
                    Text(fetched, format: .dateTime.hour().minute().second())
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.tertiary)
                }
            }

            if let metrics = serverMetrics {
                HStack(spacing: 16) {
                    if let cpu = metrics.cpuPercent {
                        serverGauge("CPU", fraction: cpu / 100,
                                    label: "\(Int(cpu))%",
                                    detail: metrics.load1.map { String(format: "load %.2f", $0) })
                    }
                    if let ram = metrics.ramUsedFraction {
                        serverGauge("RAM", fraction: ram,
                                    label: "\(Int(ram * 100))%",
                                    detail: metrics.ramUsedMiB.map { String(format: "%.1f GiB used", $0 / 1024) })
                    }
                    if let disk = metrics.diskUsedFraction {
                        serverGauge("Disk /", fraction: disk,
                                    label: "\(Int(disk * 100))%",
                                    detail: metrics.diskAvailGiB.map { String(format: "%.0f GiB free", $0) })
                    }
                    if let inKbps = metrics.netInKbps, let outKbps = metrics.netOutKbps {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Network")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text("↓ \(rateLabel(inKbps))")
                                .font(.callout.monospacedDigit())
                            Text("↑ \(rateLabel(outKbps))")
                                .font(.callout.monospacedDigit())
                        }
                    }
                    Spacer()
                }
                .padding(12)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
            } else {
                Text("Netdata on 10.0.100.232:19999 is not responding.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func serverGauge(_ title: String, fraction: Double, label: String, detail: String?) -> some View {
        HStack(spacing: 8) {
            Gauge(value: min(max(fraction, 0), 1)) {
                Text(title)
            } currentValueLabel: {
                Text(label)
                    .font(.caption.monospacedDigit())
            }
            .gaugeStyle(.accessoryCircularCapacity)
            .tint(fraction > 0.9 ? .red : fraction > 0.75 ? .orange : .green)
            .scaleEffect(0.85)

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if let detail {
                    Text(detail)
                        .font(.caption.monospacedDigit())
                }
            }
        }
    }

    private func uptimeLabel(_ seconds: Double) -> String {
        let days = Int(seconds) / 86_400
        if days > 0 { return "\(days)d" }
        let hours = Int(seconds) / 3600
        return hours > 0 ? "\(hours)h" : "\(Int(seconds) / 60)m"
    }

    private func rateLabel(_ kbps: Double) -> String {
        kbps >= 1000 ? String(format: "%.1f Mb/s", kbps / 1000) : String(format: "%.0f kb/s", kbps)
    }

    // MARK: - Render activity

    /// Timestamps inside the selected range.
    private var rangedTimestamps: [Date] {
        guard let days = range.days else { return renderTimestamps }
        let cutoff = Calendar.current.date(byAdding: .day, value: -(days - 1), to: Calendar.current.startOfDay(for: Date())) ?? .distantPast
        return renderTimestamps.filter { $0 >= cutoff }
    }

    /// Days shown in the heatmap: the range, or all history capped at a year.
    private var heatmapDayCount: Int {
        if let days = range.days { return days }
        guard let first = renderTimestamps.first else { return 30 }
        let span = (Calendar.current.dateComponents(
            [.day], from: Calendar.current.startOfDay(for: first), to: Date()).day ?? 0) + 1
        return min(max(span, 30), 366)
    }

    private var activitySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                sectionTitle("Activity")
                Spacer()
                Picker("Range", selection: $range) {
                    ForEach(HealthRange.allCases) { r in
                        Text(r.rawValue).tag(r)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
            }

            let summary = ActivityStats.summarize(timestamps: rangedTimestamps)
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 130), spacing: 12)], spacing: 12) {
                StatTile(title: "Renders", value: "\(summary.totalCount)", systemImage: "photo.stack")
                StatTile(title: "Active days", value: "\(summary.activeDays)", systemImage: "calendar")
                StatTile(title: "Current streak", value: "\(summary.currentStreak)d", systemImage: "flame")
                StatTile(title: "Longest streak", value: "\(summary.longestStreak)d", systemImage: "trophy")
                StatTile(
                    title: "Peak hour",
                    value: summary.peakHour.map { hourLabel($0) } ?? "—",
                    systemImage: "clock"
                )
            }

            ActivityHeatmapView(
                days: ActivityStats.dayCounts(timestamps: rangedTimestamps, days: heatmapDayCount)
            )
            .padding(12)
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
        }
    }

    /// "11 AM" style label for an hour of day.
    private func hourLabel(_ hour: Int) -> String {
        var components = DateComponents()
        components.hour = hour
        let date = Calendar.current.date(from: components) ?? Date()
        let formatter = DateFormatter()
        formatter.dateFormat = "h a"
        return formatter.string(from: date)
    }

    // MARK: - Watched services

    /// Datadog-style time-series trends from persisted samples (CPU, disk,
    /// service availability) over the selected range.
    private var trendsSection: some View {
        let samples = MetricsHistory.downsample(
            monitor.metricsHistory.samples(days: range.days), maxPoints: 120)
        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                sectionTitle("Trends")
                Spacer()
                Picker("", selection: $range) {
                    ForEach(HealthRange.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented).fixedSize().labelsHidden()
            }
            if samples.count < 2 {
                Text("Collecting samples… trends appear as the monitor runs.")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 300), spacing: 12)], spacing: 12) {
                    trendChart("CPU", samples: samples, unit: "%", color: .blue) { $0.cpuPercent }
                    trendChart("Disk used", samples: samples, unit: "%", color: .purple) { $0.diskUsedPercent }
                    trendChart("Services up", samples: samples, unit: "%", color: .green) { $0.serviceUpPercent }
                }
            }
        }
    }

    @ViewBuilder
    private func trendChart(_ title: String, samples: [MetricsSample], unit: String,
                            color: Color, value: @escaping (MetricsSample) -> Double?) -> some View {
        let points = samples.compactMap { s -> (Date, Double)? in value(s).map { (s.date, $0) } }
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Spacer()
                if let last = points.last {
                    Text(String(format: "%.0f%@", last.1, unit)).font(.caption.monospacedDigit())
                }
            }
            if points.count < 2 {
                Text("no data").font(.caption2).foregroundStyle(.tertiary).frame(height: 90)
            } else {
                Chart(points, id: \.0) { pt in
                    AreaMark(x: .value("t", pt.0), y: .value(unit, pt.1))
                        .foregroundStyle(color.opacity(0.15))
                    LineMark(x: .value("t", pt.0), y: .value(unit, pt.1))
                        .foregroundStyle(color)
                        .interpolationMethod(.monotone)
                }
                .chartYScale(domain: 0...100)
                .chartYAxis { AxisMarks(values: [0, 50, 100]) }
                .chartXAxis { AxisMarks(values: .automatic(desiredCount: 3)) }
                .frame(height: 90)
            }
        }
        .padding(10)
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 10))
    }

    /// Littleroundbox server health from the get_server_health tool. Containers
    /// and — always — the Suppressed Alerts section (per the 2026-07-06 handoff:
    /// a silently-muted-for-7-weeks container must never be invisible again).
    private var serverHealthSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                sectionTitle("Littleroundbox Server")
                if serverHealth.isLoading { ProgressView().controlSize(.small) }
                Spacer()
                if let ts = serverHealth.lastFetched {
                    Text(ts.formatted(date: .omitted, time: .shortened))
                        .font(.caption2).foregroundStyle(.tertiary)
                }
                Button { Task { await serverHealth.fetch() } } label: { Image(systemName: "arrow.clockwise") }
                    .controlSize(.small)
            }

            if let h = serverHealth.health {
                if let sys = h.system {
                    HStack(spacing: 14) {
                        ForEach(serverStats(sys), id: \.0) { stat in
                            Text("\(stat.0): \(stat.1)").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
                // Problem containers (running ones stay quiet).
                if !h.problemContainers.isEmpty {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 200), spacing: 10)], spacing: 10) {
                        ForEach(h.problemContainers) { c in containerCard(c) }
                    }
                }
                // Kira image-pipeline health (surfaced from the server).
                if let kira = h.kira { kiraView(kira) }
                // Suppressed alerts — ALWAYS shown (empty is a signal too).
                suppressedAlertsView(h.suppressed)
                // Mac pipeline reachability, if the server reports it.
                if let pipeline = h.macPipeline, !pipeline.isEmpty {
                    Text("Mac pipeline: " + pipeline.map { "\($0.name) \($0.isHealthy ? "✓" : "✗")" }.joined(separator: "  "))
                        .font(.caption2).foregroundStyle(.secondary)
                }
            } else {
                Text(serverHealth.lastError ?? "Server health not loaded.")
                    .font(.caption).foregroundStyle(.secondary)
                Text("Consumes coffeeshop-server's get_server_health MCP tool (in progress). Endpoint: \(serverHealth.endpoint)")
                    .font(.caption2).foregroundStyle(.tertiary)
            }
        }
    }

    private func serverStats(_ s: ServerHealthSystem) -> [(String, String)] {
        var out: [(String, String)] = []
        if let d = s.disk { out.append(("disk", d)) }
        if let m = s.memory { out.append(("mem", m)) }
        if let l = s.load { out.append(("load", l)) }
        if let u = s.uptime { out.append(("up", u)) }
        return out
    }

    private func containerCard(_ c: ServerHealthContainer) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(c.name).font(.callout.weight(.medium)).lineLimit(1)
                Spacer()
                if c.isSuppressed {
                    Text("SUPPRESSED").font(.system(size: 9, weight: .bold))
                        .padding(.horizontal, 4).padding(.vertical, 1)
                        .background(Color.orange, in: Capsule()).foregroundStyle(.white)
                }
            }
            Text(c.state ?? "unknown").font(.caption).foregroundStyle(.red)
            if c.lacksRestartPolicy {
                Label("no restart policy", systemImage: "exclamationmark.triangle.fill")
                    .font(.caption2).foregroundStyle(.orange)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(.red.opacity(0.5), lineWidth: 1.5))
    }

    private func kiraView(_ k: ServerHealthKira) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 6) {
                Image(systemName: k.isHealthy ? "checkmark.seal.fill" : "exclamationmark.triangle.fill")
                    .foregroundStyle(k.isHealthy ? Color.green : Color.orange)
                Text("Kira pipeline").font(.caption.weight(.semibold))
                if let state = k.state { Text(state).font(.caption2).foregroundStyle(.secondary) }
            }
            HStack(spacing: 14) {
                if let p = k.poolStock { Text("pool: \(p)").font(.caption2).foregroundStyle(.secondary) }
                if let b = k.lastBuild {
                    Text("last build: \(b)\(k.lastBuildStatus.map { " (\($0))" } ?? "")")
                        .font(.caption2).foregroundStyle(.secondary)
                }
                if let w = k.watchdog { Text("watchdog: \(w)").font(.caption2).foregroundStyle(.secondary) }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(8)
        .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
    }

    @ViewBuilder
    private func suppressedAlertsView(_ alerts: [ServerHealthSuppressed]) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: alerts.isEmpty ? "bell.slash" : "bell.badge.fill")
                    .foregroundStyle(alerts.isEmpty ? Color.secondary : Color.orange)
                Text("Suppressed Alerts (\(alerts.count))").font(.caption.weight(.semibold))
            }
            if alerts.isEmpty {
                Text("None — nothing is being hidden.").font(.caption2).foregroundStyle(.secondary)
            } else {
                ForEach(alerts) { a in
                    VStack(alignment: .leading, spacing: 1) {
                        Text(a.alert).font(.caption)
                        Text([a.reason, a.since].compactMap { $0 }.joined(separator: " · "))
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(6)
                    .background(.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 6))
                }
            }
        }
        .padding(8)
        .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
    }

    private var servicesSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                sectionTitle("Watched Services")
                Spacer()
                Button {
                    showAddService.toggle()
                } label: {
                    Label("Add", systemImage: "plus")
                }
                .controlSize(.small)
                .popover(isPresented: $showAddService, arrowEdge: .bottom) {
                    addServiceForm
                }
            }

            if monitor.services.isEmpty {
                Text("No services watched yet. Add a health endpoint (e.g. the Bree daemon or LM Studio) to see it here.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            } else {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 200), spacing: 12)], spacing: 12) {
                    ForEach(monitor.services) { health in
                        serviceCard(
                            service: health.service,
                            state: health.state,
                            detail: health.detail ?? health.service.urlString,
                            latencyMs: health.latencyMs,
                            uptime: monitor.uptime.stats(
                                serviceId: health.service.id,
                                days: range.days ?? 90
                            )
                        )
                        .contextMenu {
                            Button("Configure Control…") { editingControl = health.service }
                            Button("Remove", role: .destructive) {
                                removeService(health.service)
                            }
                        }
                    }
                }
            }
        }
        .sheet(item: $editingControl) { service in
            ServiceControlEditor(service: service) { updated in
                if let idx = monitor.watchedServices.firstIndex(where: { $0.id == updated.id }) {
                    monitor.watchedServices[idx] = updated
                    persistWatchedServices()
                }
                editingControl = nil
            } onCancel: {
                editingControl = nil
            }
        }
    }

    private var addServiceForm: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Watch a service")
                .font(.headline)
            TextField("Name (e.g. Bree Daemon)", text: $newServiceName)
                .textFieldStyle(.roundedBorder)
            TextField("Health URL (e.g. http://10.0.100.232:8080/health)", text: $newServiceURL)
                .textFieldStyle(.roundedBorder)
            HStack {
                Spacer()
                Button("Add") { addService() }
                    .buttonStyle(.borderedProminent)
                    .disabled(newServiceName.isEmpty || URL(string: newServiceURL)?.host == nil)
            }
        }
        .padding(16)
        .frame(width: 360)
    }

    // MARK: - Host memory

    private var memorySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                sectionTitle("This Mac")
                if let uptime = monitor.hostMetrics?.uptimeSeconds {
                    Text("up \(uptimeLabel(uptime))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            HStack(spacing: 16) {
                if let host = monitor.hostMetrics, let cpu = host.cpuPercent {
                    serverGauge("CPU", fraction: cpu / 100,
                                label: "\(Int(cpu))%",
                                detail: host.load1.map { String(format: "load %.2f", $0) })
                }
                if let fraction = monitor.memoryUsedFraction,
                   let used = monitor.memoryUsedGB,
                   let total = monitor.memoryTotalGB {
                    serverGauge("Memory", fraction: fraction,
                                label: "\(Int(fraction * 100))%",
                                detail: String(format: "%.1f / %.0f GB", used, total))
                }
                if let host = monitor.hostMetrics,
                   let disk = host.diskUsedFraction, let free = host.diskFreeGB {
                    serverGauge("Disk", fraction: disk,
                                label: "\(Int(disk * 100))%",
                                detail: String(format: "%.0f GB free", free))
                }
                if monitor.hostMetrics == nil && monitor.memoryUsedFraction == nil {
                    Text("Host stats unavailable")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(12)
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
        }
    }

    // MARK: - Events

    private var eventsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionTitle("Events")
            if monitor.events.isEmpty {
                Text("No events yet — state changes appear here.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            } else {
                VStack(spacing: 0) {
                    ForEach(monitor.events) { event in
                        HStack(alignment: .firstTextBaseline, spacing: 8) {
                            Image(systemName: severityIcon(event.severity))
                                .foregroundStyle(severityColor(event.severity))
                                .font(.caption)
                            Text(event.timestamp, format: .dateTime.hour().minute().second())
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                            Text(event.message)
                                .font(.callout)
                                .lineLimit(2)
                            Spacer()
                        }
                        .padding(.vertical, 5)
                        .padding(.horizontal, 10)
                        if event.id != monitor.events.last?.id {
                            Divider()
                        }
                    }
                }
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
            }
        }
    }

    // MARK: - Pieces

    private func sectionTitle(_ title: String) -> some View {
        Text(title)
            .font(.headline)
    }

    private func statusBadge(_ state: HealthState, font: Font = .callout) -> some View {
        Label(state.label, systemImage: state.systemImage)
            .font(font.weight(.semibold))
            .foregroundStyle(stateColor(state))
    }

    private func serviceCard(
        service: WatchedService, state: HealthState, detail: String?, latencyMs: Int?,
        uptime: UptimeStats? = nil
    ) -> some View {
        let isBusy = busyServiceId == service.id
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(service.name)
                    .font(.callout.weight(.medium))
                    .lineLimit(1)
                Spacer()
                if isBusy {
                    ProgressView().controlSize(.small)
                } else if let latencyMs {
                    Text("\(latencyMs) ms")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            statusBadge(state, font: .caption)
            if let detail, !detail.isEmpty {
                Text(detail)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            if let uptime, uptime.totalChecks > 0 {
                Text(String(
                    format: "%.1f%% up · %d interruption%@ (%@)",
                    uptime.uptimePercent,
                    uptime.interruptions,
                    uptime.interruptions == 1 ? "" : "s",
                    range.days.map { "\($0)d" } ?? "90d"
                ))
                .font(.caption2.monospacedDigit())
                .foregroundStyle(uptime.interruptions > 0 ? .orange : .secondary)
            }

            // Control buttons (only when a control is configured).
            if let control = service.control, control.isActionable {
                HStack(spacing: 6) {
                    Button { performControl(.restart, on: service) } label: {
                        Label("Restart", systemImage: "arrow.clockwise").labelStyle(.iconOnly)
                    }.help("Restart")
                    Button { performControl(.start, on: service) } label: {
                        Label("Start", systemImage: "play.fill").labelStyle(.iconOnly)
                    }.help("Start")
                    Button { performControl(.stop, on: service) } label: {
                        Label("Stop", systemImage: "stop.fill").labelStyle(.iconOnly)
                    }.help("Stop")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(isBusy)
            }
            if let toast = controlToast, toast.service == service.name {
                Text(toast.message)
                    .font(.caption2)
                    .foregroundStyle(toast.isError ? .orange : .green)
                    .lineLimit(2)
            }
        }
        .frame(maxWidth: .infinity, minHeight: 64, alignment: .leading)
        .padding(12)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(stateColor(state).opacity(state == .healthy ? 0 : 0.5), lineWidth: 1.5)
        )
    }

    private func performControl(_ action: ServiceAction, on service: WatchedService) {
        busyServiceId = service.id
        controlToast = nil
        Task {
            defer { busyServiceId = nil }
            do {
                _ = try await controller.perform(action, on: service)
                controlToast = (service.name, "\(action.rawValue.capitalized) ok", false)
                // Re-probe shortly so the badge reflects the new state.
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                await monitor.checkNow()
            } catch {
                controlToast = (service.name, error.localizedDescription, true)
            }
        }
    }

    private func stateColor(_ state: HealthState) -> Color {
        switch state {
        case .healthy: return .green
        case .degraded: return .orange
        case .down: return .red
        case .unknown: return .gray
        }
    }

    private func severityIcon(_ severity: HealthEvent.Severity) -> String {
        switch severity {
        case .info: return "checkmark.circle.fill"
        case .warning: return "exclamationmark.triangle.fill"
        case .error: return "xmark.octagon.fill"
        }
    }

    private func severityColor(_ severity: HealthEvent.Severity) -> Color {
        switch severity {
        case .info: return .green
        case .warning: return .orange
        case .error: return .red
        }
    }

    // MARK: - Service management

    private func addService() {
        let service = WatchedService(name: newServiceName, urlString: newServiceURL)
        monitor.watchedServices.append(service)
        persistWatchedServices()
        newServiceName = ""
        newServiceURL = ""
        showAddService = false
        Task { await monitor.checkNow() }
    }

    private func removeService(_ service: WatchedService) {
        monitor.watchedServices.removeAll { $0.id == service.id }
        persistWatchedServices()
        Task { await monitor.checkNow() }
    }

    private func persistWatchedServices() {
        var settings = DesktopSettings.load()
        settings.watchedServices = monitor.watchedServices
        settings.save()
    }
}
