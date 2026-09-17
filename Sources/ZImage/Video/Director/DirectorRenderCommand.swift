// DirectorRenderCommand.swift — the testable half of `comfybox director-render`
// (docs/FDD-ltx-director-tab.md §4.3, WP4).
//
// The CLI is a thin HTTP client of the RUNNING server: it reads a
// `.cbdirector` file, POSTs it to /v1/video/director (or /validate) and, with
// --wait, polls /v1/video/status/{id}. It never loads weights and never starts
// an engine. Argument parsing and the request/poll flow live here, in the
// ZImage library, so ZImageTests can drive them with a stub transport; the
// executable target only wires stdout/stderr, a real WarmServerClient and
// process exit.

import Foundation

public enum DirectorRenderArgsError: Error, Equatable, LocalizedError {
  case usage(String)
  case helpRequested

  public var errorDescription: String? {
    switch self {
    case .usage(let message): return message
    case .helpRequested: return DirectorRenderArgs.usage
    }
  }
}

public struct DirectorRenderArgs: Equatable, Sendable {
  public static let defaultServer = "http://127.0.0.1:7870"

  public var filePath: String
  /// As given (default http://127.0.0.1:7870); `host`/`port` are parsed from it.
  public var server: String
  public var host: String
  public var port: UInt16
  public var outputPath: String?
  public var source: String
  public var validateOnly: Bool
  public var wait: Bool
  public var pollSeconds: Double

  public static let usage = """
    Usage: comfybox director-render <file.cbdirector> [options]

    Submits a Director timeline to a RUNNING ComfyBox server (it never renders
    in-process and never starts an engine).

    Options:
      --server <url>        Server base URL (default \(defaultServer))
      --output <path>       Output .mp4 name or path under the server's output directory
                            (default director-<session>.mp4)
      --source <name>       Queue attribution (default cli)
      --validate-only       POST /v1/video/director/validate and print issues + plan;
                            exits 1 when the timeline has errors
      --wait                Poll /v1/video/status/{id} until the job finishes;
                            exits 1 when it fails
      --poll-seconds <n>    Poll interval for --wait (default 2)
      -h, --help            Show this help
    """

  /// Pure: no file or network access (the file is read by the client).
  public static func parse(_ argv: [String]) throws -> DirectorRenderArgs {
    var filePath: String?
    var server = defaultServer
    var outputPath: String?
    var source = "cli"
    var validateOnly = false
    var wait = false
    var pollSeconds = 2.0

    var iterator = argv.makeIterator()
    func value(for flag: String) throws -> String {
      guard let v = iterator.next() else {
        throw DirectorRenderArgsError.usage("\(flag) requires a value")
      }
      return v
    }
    while let arg = iterator.next() {
      switch arg {
      case "-h", "--help":
        throw DirectorRenderArgsError.helpRequested
      case "--server":
        server = try value(for: arg)
      case "--output", "-o":
        outputPath = try value(for: arg)
      case "--source":
        source = try value(for: arg)
      case "--validate-only":
        validateOnly = true
      case "--wait":
        wait = true
      case "--poll-seconds":
        let raw = try value(for: arg)
        guard let n = Double(raw), n.isFinite, n > 0 else {
          throw DirectorRenderArgsError.usage("--poll-seconds must be a positive number (got \(raw))")
        }
        pollSeconds = n
      default:
        if arg.hasPrefix("-") {
          throw DirectorRenderArgsError.usage("Unknown option: \(arg)")
        }
        if let existing = filePath {
          throw DirectorRenderArgsError.usage(
            "Only one <file.cbdirector> is accepted (got \(existing) and \(arg))")
        }
        filePath = arg
      }
    }

    guard let filePath else {
      throw DirectorRenderArgsError.usage("Missing <file.cbdirector>")
    }
    if validateOnly && wait {
      throw DirectorRenderArgsError.usage("--wait cannot be combined with --validate-only")
    }
    let (host, port) = try parseServer(server)
    return DirectorRenderArgs(
      filePath: filePath, server: server, host: host, port: port, outputPath: outputPath,
      source: source, validateOnly: validateOnly, wait: wait, pollSeconds: pollSeconds)
  }

  /// `http://host[:port][/]` only: WarmServerClient speaks plain HTTP and
  /// appends absolute /v1 paths to the base, so a scheme other than http or a
  /// base path would silently misroute.
  private static func parseServer(_ raw: String) throws -> (String, UInt16) {
    let bad = DirectorRenderArgsError.usage(
      "--server must look like http://host[:port] (got \(raw))")
    guard let comps = URLComponents(string: raw), comps.scheme?.lowercased() == "http",
      let host = comps.host, !host.isEmpty,
      comps.path.isEmpty || comps.path == "/", comps.query == nil
    else { throw bad }
    guard let port = comps.port else { return (host, 80) }
    guard (1...65535).contains(port) else { throw bad }
    return (host, UInt16(port))
  }
}

public enum DirectorRenderClient {
  /// BSD EX_USAGE: bad arguments or a timeline file this build cannot read.
  public static let usageExitCode: Int32 = 64

  private struct Envelope: Encodable {
    let timeline: DirectorTimeline
    let outputPath: String?
    let source: String?
  }

  /// The request body: `{timeline, output_path?, source}` for a render,
  /// `{timeline}` for --validate-only. snake_case via DirectorJSON.
  public static func envelope(timeline: DirectorTimeline, args: DirectorRenderArgs) throws -> Data {
    let env = args.validateOnly
      ? Envelope(timeline: timeline, outputPath: nil, source: nil)
      : Envelope(timeline: timeline, outputPath: args.outputPath, source: args.source)
    return try DirectorJSON.encoder(pretty: false).encode(env)
  }

  /// One human line for a status payload: "processing · chunk 2/3 · 40%",
  /// "processing · stitching · 97%" (stage_index == stage_count), "queued".
  public static func progressLine(_ status: [String: Any]) -> String {
    var parts = [status["status"] as? String ?? "unknown"]
    if let index = status["stage_index"] as? Int, let count = status["stage_count"] as? Int, count > 0 {
      parts.append(index >= count ? "stitching" : "chunk \(index + 1)/\(count)")
    }
    if let pct = status["progress_percent"] as? Int {
      parts.append("\(pct)%")
    }
    return parts.joined(separator: " · ")
  }

  /// Run the command against `transport`; returns the process exit code.
  /// `out` gets response JSON and progress lines, `err` gets failures.
  public static func run(
    args: DirectorRenderArgs,
    transport: WarmServerTransport,
    out: @Sendable (String) -> Void,
    err: @Sendable (String) -> Void,
    sleep: @Sendable (Double) async throws -> Void = { try await Task.sleep(nanoseconds: UInt64($0 * 1_000_000_000)) }
  ) async -> Int32 {
    let timeline: DirectorTimeline
    do {
      timeline = try DirectorDocument.read(from: URL(fileURLWithPath: (args.filePath as NSString).expandingTildeInPath))
    } catch DirectorError.unsupportedVersion(let v) {
      err("\(args.filePath): timeline version \(v) is newer than this build supports (version 1)")
      return usageExitCode
    } catch {
      err("\(error.localizedDescription)")
      return 1
    }

    let body: Data
    do {
      body = try envelope(timeline: timeline, args: args)
    } catch {
      err("could not encode the timeline: \(error)")
      return 1
    }

    let path = args.validateOnly ? "/v1/video/director/validate" : "/v1/video/director"
    let status: Int
    let data: Data
    do {
      (status, data) = try await transport.post(path, body: body)
    } catch {
      err("cannot reach \(args.server)\(path): \(error.localizedDescription)")
      return 1
    }
    let text = String(data: data, encoding: .utf8) ?? ""
    let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]

    if args.validateOnly {
      guard status == 200 else {
        err("validate failed (HTTP \(status)): \(text)")
        return 1
      }
      out(text)
      return (json?["ok"] as? Bool) == true ? 0 : 1
    }

    guard status == 202 else {
      err("director submit refused (HTTP \(status)): \(text)")
      return 1
    }
    out(text)
    guard args.wait else { return 0 }
    guard let jobId = json?["job_id"] as? String, !jobId.isEmpty else {
      err("202 without a job_id; cannot --wait")
      return 1
    }

    var lastLine = ""
    while true {
      do {
        try await sleep(args.pollSeconds)
      } catch {
        err("interrupted while waiting for \(jobId)")
        return 1
      }
      let pollStatus: Int
      let pollData: Data
      do {
        (pollStatus, pollData) = try await transport.get("/v1/video/status/\(jobId)")
      } catch {
        err("status poll failed for \(jobId): \(error.localizedDescription)")
        return 1
      }
      let pollText = String(data: pollData, encoding: .utf8) ?? ""
      // A non-200 poll ends the wait immediately: local video jobs are
      // non-durable, so a 404 after an engine restart will never recover.
      guard pollStatus == 200,
        let job = (try? JSONSerialization.jsonObject(with: pollData)) as? [String: Any]
      else {
        err("status poll for \(jobId) returned HTTP \(pollStatus): \(pollText)")
        return 1
      }
      switch job["status"] as? String {
      case "succeeded":
        out(pollText)
        return 0
      case "failed":
        let message = job["error"] as? String ?? "unknown error"
        let interrupted = (job["interrupted"] as? Bool) == true ? " (interrupted)" : ""
        err("director job \(jobId) failed\(interrupted): \(message)")
        return 1
      default:
        let line = progressLine(job)
        if line != lastLine {
          out(line)
          lastLine = line
        }
      }
    }
  }
}
