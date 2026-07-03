// NetdataClientTests.swift — Netdata allmetrics parsing

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("NetdataClient")
struct NetdataClientTests {
    /// Trimmed live shape from littleroundbox's Netdata v2.10.3.
    private static let sample = #"""
    {
      "system.cpu": {
        "name": "system.cpu", "units": "percentage",
        "dimensions": {
          "user": {"name": "user", "value": 16.7958656},
          "system": {"name": "system", "value": 6.7183463},
          "idle": {"name": "idle", "value": 75.9689922},
          "iowait": {"name": "iowait", "value": 0.2583979}
        }
      },
      "system.ram": {
        "name": "system.ram", "units": "MiB",
        "dimensions": {
          "free": {"name": "free", "value": 728.7421875},
          "used": {"name": "used", "value": 6209.2304688},
          "cached": {"name": "cached", "value": 8334.0117188},
          "buffers": {"name": "buffers", "value": 696.375}
        }
      },
      "system.load": {
        "name": "system.load", "units": "load",
        "dimensions": {"load1": {"name": "load1", "value": 1.43}}
      },
      "system.net": {
        "name": "system.net", "units": "kilobits/s",
        "dimensions": {
          "InOctets": {"name": "received", "value": 19.6451417},
          "OutOctets": {"name": "sent", "value": -20.0722086}
        }
      },
      "system.uptime": {
        "name": "system.uptime", "units": "seconds",
        "dimensions": {"uptime": {"name": "uptime", "value": 4268406.24}}
      },
      "disk_space./": {
        "name": "disk_space./", "units": "GiB",
        "dimensions": {
          "avail": {"name": "avail", "value": 30.09235},
          "used": {"name": "used", "value": 178.4860954},
          "reserved_for_root": {"name": "reserved for root", "value": 11.2354088}
        }
      }
    }
    """#

    @Test("parses the live allmetrics shape")
    func parsesLiveShape() throws {
        let metrics = try #require(NetdataClient.parse(allMetrics: Data(Self.sample.utf8)))
        #expect(abs((metrics.cpuPercent ?? 0) - 24.03) < 0.1)     // 100 - idle
        #expect(metrics.load1 == 1.43)
        #expect(abs((metrics.ramUsedMiB ?? 0) - 6209.23) < 0.1)
        #expect(abs((metrics.ramTotalMiB ?? 0) - 15968.36) < 0.1) // sum of dims
        #expect(abs((metrics.diskUsedGiB ?? 0) - 178.49) < 0.1)
        #expect(metrics.netOutKbps ?? -1 > 0)                     // magnitude of negative egress
        #expect(metrics.uptimeSeconds ?? 0 > 4_000_000)
    }

    @Test("derived fractions")
    func fractions() throws {
        let metrics = try #require(NetdataClient.parse(allMetrics: Data(Self.sample.utf8)))
        let ram = try #require(metrics.ramUsedFraction)
        #expect(ram > 0.35 && ram < 0.45)
        let disk = try #require(metrics.diskUsedFraction)
        #expect(disk > 0.80 && disk < 0.90)   // 178.5 / (178.5 + 30.1)
    }

    @Test("unrecognizable payload returns nil")
    func garbage() {
        #expect(NetdataClient.parse(allMetrics: Data("{}".utf8)) == nil)
        #expect(NetdataClient.parse(allMetrics: Data("not json".utf8)) == nil)
        #expect(NetdataClient.parse(allMetrics: Data(#"{"other.chart": {"dimensions": {}}}"#.utf8)) == nil)
    }
}
