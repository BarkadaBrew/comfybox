// LTX2ImageRecipe.swift — the native one-frame LTX-2 recipe contract.
//
// LTX image generation shares the video pipeline's denoising loop, which
// implements the four Euler-family modes below. Image generation always uses
// LTX's resolution/token-shifted flow sigmas; no second sigma family is wired.

import Foundation

public enum LTX2ImageRecipe {
  public static let defaultSampler = "euler"
  public static let defaultSigmaSchedule = "flow"

  public static let samplerNames = [
    "euler",
    "euler_ancestral",
    "euler_cfg_pp",
    "euler_ancestral_cfg_pp",
  ]

  public static let sigmaScheduleNames = [defaultSigmaSchedule]
  public static let engineNames = ["ltx2", "ltx-2", "ltx2-image"]

  public static func isEngineName(_ value: String?) -> Bool {
    guard let value = normalized(value) else { return false }
    return engineNames.contains(value)
  }

  public static func normalized(_ value: String?) -> String? {
    guard let value = value?.trimmingCharacters(in: .whitespacesAndNewlines),
          !value.isEmpty else { return nil }
    return value.lowercased()
  }

  public static func supportsSampler(_ value: String?) -> Bool {
    guard let value = normalized(value) else { return true }
    return samplerNames.contains(value)
  }

  public static func supportsSigmaSchedule(_ value: String?) -> Bool {
    guard let value = normalized(value) else { return true }
    return sigmaScheduleNames.contains(value)
  }

  /// A user-facing validation message shared by the server and Desktop.
  public static func validationError(sampler: String?, sigmaSchedule: String?) -> String? {
    if let sampler = normalized(sampler), !samplerNames.contains(sampler) {
      return "Native LTX-2 image sampler '\(sampler)' is unsupported; expected one of: "
        + samplerNames.joined(separator: ", ")
    }
    if let sigmaSchedule = normalized(sigmaSchedule),
       !sigmaScheduleNames.contains(sigmaSchedule) {
      return "Native LTX-2 image generation uses the shifted flow schedule; got '\(sigmaSchedule)'"
    }
    return nil
  }
}
