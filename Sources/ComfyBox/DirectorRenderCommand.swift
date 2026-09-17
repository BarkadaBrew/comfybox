// DirectorRenderCommand.swift — `comfybox director-render <file.cbdirector>`
// (docs/FDD-ltx-director-tab.md §4.3, WP4).
//
// A thin HTTP client of the RUNNING server: parsing and the submit/poll flow
// live in ZImage (`DirectorRenderArgs`, `DirectorRenderClient`) where they are
// unit-tested; this file only wires a real WarmServerClient, stdout/stderr and
// the process exit code. It never loads weights and never starts an engine.

import Foundation
import ZImage

private final class ExitCodeBox: @unchecked Sendable {
  private let lock = NSLock()
  private var _value: Int32 = 1
  var value: Int32 {
    get { lock.lock(); defer { lock.unlock() }; return _value }
    set { lock.lock(); defer { lock.unlock() }; _value = newValue }
  }
}

extension ZImageCLI {
  static func runDirectorRender(args: [String]) throws {
    let parsed: DirectorRenderArgs
    do {
      parsed = try DirectorRenderArgs.parse(args)
    } catch DirectorRenderArgsError.helpRequested {
      print(DirectorRenderArgs.usage)
      return
    } catch DirectorRenderArgsError.usage(let message) {
      fputs("Error: \(message)\n\n\(DirectorRenderArgs.usage)\n", stderr)
      exit(DirectorRenderClient.usageExitCode)
    }

    let transport = WarmServerClient(host: parsed.host, port: parsed.port)
    let code = ExitCodeBox()
    let semaphore = DispatchSemaphore(value: 0)
    Task {
      code.value = await DirectorRenderClient.run(
        args: parsed,
        transport: transport,
        out: { line in
          print(line)
          fflush(stdout)
        },
        err: { line in
          fputs("\(line)\n", stderr)
        })
      semaphore.signal()
    }
    semaphore.wait()
    if code.value != 0 {
      exit(code.value)
    }
  }
}
