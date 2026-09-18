// aperture.swift — lip-sync measurement that cannot be fooled by head motion.
//
// The pixel-difference metric this replaces counted the one-time transient of
// a subject raising her head from the opening keyframe as the voice starts,
// and reported it as lip sync. It produced two published conclusions that were
// both wrong.
//
// This measures MOUTH APERTURE from Vision's face landmarks: the mean vertical
// separation between the inner upper and inner lower lip, normalised by the
// face's own bounding-box height so it survives the subject moving toward or
// away from camera. A head that turns, nods or rises does not change aperture;
// only an opening mouth does.
//
// Usage: swift aperture.swift <video.mp4>   ->  one normalised aperture per frame

import AVFoundation
import Foundation
import Vision

let args = CommandLine.arguments
guard args.count > 1 else { FileHandle.standardError.write(Data("usage: aperture.swift <video>\n".utf8)); exit(2) }
let url = URL(fileURLWithPath: args[1])

let asset = AVURLAsset(url: url)
guard let track = asset.tracks(withMediaType: .video).first else {
    FileHandle.standardError.write(Data("no video track\n".utf8)); exit(2)
}
let reader = try AVAssetReader(asset: asset)
let output = AVAssetReaderTrackOutput(
    track: track,
    outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
output.alwaysCopiesSampleData = false
reader.add(output)
reader.startReading()

var frame = 0
while let sample = output.copyNextSampleBuffer() {
    defer { frame += 1 }
    guard let pixels = CMSampleBufferGetImageBuffer(sample) else { print("\(frame) nan"); continue }
    let request = VNDetectFaceLandmarksRequest()
    let handler = VNImageRequestHandler(cvPixelBuffer: pixels, orientation: .up, options: [:])
    do { try handler.perform([request]) } catch { print("\(frame) nan"); continue }
    guard let face = (request.results)?.first,
          let inner = face.landmarks?.innerLips
    else { print("\(frame) nan"); continue }

    // innerLips is a closed contour in face-normalised coordinates. Its
    // vertical extent IS the aperture; dividing by the face box height makes
    // it independent of how large the face is in frame.
    let points = inner.normalizedPoints
    guard points.count >= 4 else { print("\(frame) nan"); continue }
    let ys = points.map { $0.y }
    let aperture = Double(ys.max()! - ys.min()!) * Double(face.boundingBox.height)
    print("\(frame) \(aperture)")
}
