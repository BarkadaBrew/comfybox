import Testing
@testable import ComfyBoxDesktop

@Suite("VideoRequest audio")
struct VideoRequestAudioTests {
    @Test("the desktop video body carries the audio flag, on by default, off when the toggle is off")
    func audioFlagOnTheWire() {
        let on = EngineService.VideoRequest(prompt: "p", outputPath: "/tmp/x.mp4")
        #expect(on.audio == true)
        let bodyOn = EngineService.videoRequestBody(on, forceLocal: true)
        #expect(bodyOn["audio"] as? Bool == true)
        #expect(bodyOn["source"] as? String == "desktop")

        let off = EngineService.VideoRequest(prompt: "p", outputPath: "/tmp/x.mp4", audio: false)
        let bodyOff = EngineService.videoRequestBody(off, forceLocal: true)
        #expect(bodyOff["audio"] as? Bool == false)
    }
}
