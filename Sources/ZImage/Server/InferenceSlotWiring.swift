// InferenceSlotWiring.swift — the production glue between the inference slot
// table, the inference hold and the #1479 LTX-2 preemption state.
//
// FDD-glimmer-gpu-slot §3.1. Lives in one place so WarmServer and the DEBUG
// queue probe run the same closures (Codex review of #465, finding 2: the
// probe used to hand-roll `hold.request()`, so the flag/signal/watchdog path
// was untested).

import Foundation
import Logging

enum InferenceSlotWiring {

  /// Last slot released or expired: withdraw a hold request the video never
  /// claimed (it would otherwise strand the preemption flag and leave the
  /// signal raised), then wake any render that parked behind the slots.
  static func onEmpty(
    hold: InferenceHold, signal: PreemptionSignal, inFlight: LockedFlag,
    wake: @escaping @Sendable () -> Void
  ) -> @Sendable () -> Void {
    {
      if hold.cancelIfRequested() {
        signal.clear()
        inFlight.clear()
      }
      wake()
    }
  }

  /// The route context `/v1/queue/inference-slot` handlers read.
  /// - Parameter watchdogSec: how long an in-flight video gets to yield before
  ///   the request is withdrawn (the slots then wait for the render to end).
  static func routeContext(
    table: InferenceSlotTable, hold: InferenceHold,
    signal: PreemptionSignal, inFlight: LockedFlag,
    gpuBusy: @escaping @Sendable () -> Bool,
    videoRendering: @escaping @Sendable () -> Bool,
    progress: @escaping @Sendable () -> (Int?, Date?),
    watchdogSec: @escaping @Sendable () -> TimeInterval,
    logger: Logger
  ) -> InferenceSlotRouteContext {
    InferenceSlotRouteContext(
      table: table, hold: hold,
      gpuBusy: gpuBusy,
      videoRendering: videoRendering,
      progress: progress,
      raiseHold: {
        // One preemption at a time: an image job's #1479 episode already owns it.
        guard inFlight.trySet() else { return }
        guard hold.request() else { inFlight.clear(); return }
        guard videoRendering() else {
          _ = hold.cancelIfRequested()
          inFlight.clear()
          return
        }
        logger.info("inference slot: asking the in-flight LTX-2 video to checkpoint (top priority)")
        signal.raise()
        let windowSec = watchdogSec()
        DispatchQueue.global().asyncAfter(deadline: .now() + windowSec) {
          if hold.cancelIfRequested() {
            signal.clear()
            inFlight.clear()
            logger.warning("inference slot: video did not yield within \(Int(windowSec))s — slots wait for the render to finish")
          }
        }
      })
  }
}
