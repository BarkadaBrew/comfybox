// AgentServiceTests.swift — Agent request assembly + response parsing

import Testing
import Foundation
@testable import ComfyBoxDesktop
import ZImage

@Suite("AgentService")
struct AgentServiceTests {
    @Test("brief covers LTX-2.3 video: distilled cfg 1.0, NAG-live negative, motion envelope, one quoted line")
    func briefCoversLTX() {
        let p = AgentService.systemPrompt
        #expect(p.contains("LTX-2.3 VIDEO"))
        #expect(p.contains("PinkCherry"))
        #expect(p.contains("guidance is 1.0"))
        #expect(p.contains("NEGATIVE IS LIVE ANYWAY"))
        #expect(p.contains("MOTION ENVELOPE"))
        #expect(p.contains("exactly ONE short quoted line"))
        #expect(p.contains("VIDEO RECIPE"))
        #expect(p.contains("0.9953"), "the validated 10-step schedule is in the brief")
    }

    @Test("snapshot lists LTX video presets separately from image presets")
    func snapshotListsVideoPresets() {
        let s = AgentService.buildStackContext(
            model: "krea2-raw", family: "krea2", samplers: ["euler"], loras: [],
            presets: [(id: "krea-kira", model: "krea2-raw", sampler: "res_2s", steps: 12, guidance: 1.0, loras: [])],
            videoPresets: [(id: "kira-video-apple", negativeTerms: 40, loras: 0)])
        #expect(s.contains("Image presets"))
        #expect(s.contains("LTX-2.3 video presets"))
        #expect(s.contains("kira-video-apple: 40-term negative, 0 LoRAs"))
    }

    @Test("preferredProvider: the assistant slot wins, prompt optimization is the fallback")
    func preferredProvider() {
        let opt = AIProviderEndpoint(baseUrl: "http://127.0.0.1:11434/v1", model: "dolphin")
        let glimmer = AIProviderEndpoint(baseUrl: "http://127.0.0.1:11234/v1", model: "Muse-Glimmer-30B-heretic-MLX-Q6")
        #expect(AgentService.preferredProvider(AIProviderRegistry(promptOptimization: opt, assistant: glimmer)) == glimmer)
        #expect(AgentService.preferredProvider(AIProviderRegistry(promptOptimization: opt)) == opt)
        #expect(AgentService.preferredProvider(AIProviderRegistry()) == nil)
    }

    @Test("system prompt is the stack-aware expert brief (Todd 2026-09-15)")
    func systemPromptBrief() {
        let p = AgentService.systemPrompt
        #expect(p.contains("krea2-raw"))
        #expect(p.contains("guidance EXACTLY 1.0"), "accelerated lane pins guidance 1.0")
        #expect(p.contains("NEGATIVE PROMPT IS INERT"), "negatives inert without CFG")
        #expect(p.contains("3.0–4.0"), "CFG-on lane guidance band")
        #expect(p.contains("FINAL PROMPT"), "conversation converges on a formatted prompt")
        #expect(p.contains("entirely uncensored"))
        #expect(!p.contains("Z-Image / Flux generation stack"), "the stale Z-Image brief is gone")
    }

    @Test("live stack snapshot rides as a second system message, before history")
    func stackContextMessage() {
        let history = [AgentMessage(role: .user, text: "hi")]
        let body = AgentService.requestBody(model: "m", messages: history, context: "LIVE STACK SNAPSHOT: x")
        let msgs = body["messages"] as! [[String: String]]
        #expect(msgs.count == 3)
        #expect(msgs[0]["role"] == "system" && msgs[0]["content"] == AgentService.systemPrompt)
        #expect(msgs[1]["role"] == "system" && msgs[1]["content"]?.hasPrefix("LIVE STACK SNAPSHOT") == true)
        #expect(msgs[2]["role"] == "user")
        let without = AgentService.requestBody(model: "m", messages: history)
        #expect((without["messages"] as! [[String: String]]).count == 2, "no snapshot → no extra message")
    }

    @Test("buildStackContext lists model, samplers, LoRAs by category and preset recipes")
    func stackContextBuilder() {
        let text = AgentService.buildStackContext(
            model: "krea2-raw", family: "krea2", samplers: ["euler", "res_2s", "res_3s"],
            loras: [(filename: "kroma-v0.3.safetensors", category: "style"), (filename: "krea2_turbo_distill_r256.safetensors", category: "accelerator")],
            presets: [(id: "krea-kira-sfw", model: "krea2-raw", sampler: "euler", steps: 8, guidance: 1.0, loras: ["krea2_turbo_distill_r256.safetensors=0.6"])])
        #expect(text.hasPrefix("LIVE STACK SNAPSHOT"))
        #expect(text.contains("Loaded model: krea2-raw (family krea2)"))
        #expect(text.contains("euler, res_2s, res_3s"))
        #expect(text.contains("accelerator: krea2_turbo_distill_r256.safetensors"))
        #expect(text.contains("krea-kira-sfw: model krea2-raw; sampler euler; steps 8; guidance 1.0; loras [krea2_turbo_distill_r256.safetensors=0.6]"))
    }

    @Test("request body prepends the system prompt and maps history")
    func requestBody() {
        let history = [
            AgentMessage(role: .user, text: "make a moody portrait"),
            AgentMessage(role: .assistant, text: "sure, here are ideas"),
            AgentMessage(role: .user, text: "add rim light"),
        ]
        let body = AgentService.requestBody(model: "dans-pe-v1.3.0-24b-heresy@8bit", messages: history)
        #expect(body["model"] as? String == "dans-pe-v1.3.0-24b-heresy@8bit")
        #expect(body["stream"] as? Bool == false)
        let msgs = body["messages"] as! [[String: String]]
        #expect(msgs.count == 4)                       // system + 3
        #expect(msgs.first?["role"] == "system")
        #expect(msgs[1]["role"] == "user")
        #expect(msgs[1]["content"] == "make a moody portrait")
        #expect(msgs.last?["content"] == "add rim light")
    }

    @Test("parseReply pulls assistant content from the OpenAI shape")
    func parseReply() {
        let json = #"""
        {"choices": [{"message": {"role": "assistant", "content": "  Try this.\nPROMPT: a moody portrait  "}}]}
        """#
        #expect(AgentService.parseReply(Data(json.utf8)) == "Try this.\nPROMPT: a moody portrait")
        #expect(AgentService.parseReply(Data("garbage".utf8)) == nil)
        #expect(AgentService.parseReply(Data(#"{"choices":[]}"#.utf8)) == nil)
    }

    @Test("suggestedPrompt extracts a PROMPT: line, case-insensitive")
    func suggestedPrompt() {
        let text = "Here's an idea for you.\nprompt: kira making espresso, warm morning light\nHope that helps!"
        #expect(AgentService.suggestedPrompt(from: text) == "kira making espresso, warm morning light")
        // No marker -> nil (caller can fall back to the whole message).
        #expect(AgentService.suggestedPrompt(from: "just chatting") == nil)
        // Empty value after marker -> nil.
        #expect(AgentService.suggestedPrompt(from: "PROMPT:   ") == nil)
    }

    @Test("system prompt instructs the PROMPT: convention and json action")
    func systemPrompt() {
        #expect(AgentService.systemPrompt.contains("PROMPT:"))
        #expect(AgentService.systemPrompt.contains("json"))
    }

    @Test("parseAction reads a fenced json action block")
    func parseActionFenced() {
        let reply = """
        Here's a moody setup.
        ```json
        {"prompt": "kira at golden hour, 85mm", "negative_prompt": "blurry", "steps": 9, "guidance": 3.5, "width": 1024, "height": 1536, "seed": 42, "loras": ["moody=0.7"], "generate": true}
        ```
        """
        let action = AgentService.parseAction(from: reply)
        #expect(action != nil)
        #expect(action?.prompt == "kira at golden hour, 85mm")
        #expect(action?.negativePrompt == "blurry")
        #expect(action?.steps == 9)
        #expect(action?.guidance == 3.5)
        #expect(action?.width == 1024)
        #expect(action?.height == 1536)
        #expect(action?.seed == 42)
        #expect(action?.loras == ["moody=0.7"])
        #expect(action?.generate == true)
    }

    @Test("parseAction reads a bare json object and partial keys")
    func parseActionBare() {
        let action = AgentService.parseAction(from: #"Sure. {"steps": 12, "guidance": 4}"#)
        #expect(action?.steps == 12)
        #expect(action?.guidance == 4.0)
        #expect(action?.prompt == nil)
        #expect(action?.width == nil)
    }

    @Test("parseAction returns nil for no block or empty changes")
    func parseActionNil() {
        #expect(AgentService.parseAction(from: "just prose, no json") == nil)
        #expect(AgentService.parseAction(from: "```json\n{}\n```") == nil)
        #expect(AgentService.parseAction(from: #"{"prompt": ""}"#) == nil)
    }

    @Test("action summary describes the changes")
    func actionSummary() {
        var action = AgentAction()
        action.steps = 9
        action.width = 1024; action.height = 1536
        action.generate = true
        let s = action.summary
        #expect(s.contains("steps 9"))
        #expect(s.contains("1024×1536"))
        #expect(s.contains("generate"))
    }

    // MARK: - Studio Pack task cards (FR-6 / #199)

    @Test("parseAction reads studio_pack_id, template_id, and model")
    func parseActionStudioPackFields() {
        let reply = #"""
        ```json
        {"studio_pack_id": "life-design-healthcare", "template_id": "cpr", "model": "z-image-turbo"}
        ```
        """#
        let action = AgentService.parseAction(from: reply)
        #expect(action?.studioPackId == "life-design-healthcare")
        #expect(action?.templateId == "cpr")
        #expect(action?.model == "z-image-turbo")
    }

    @Test("parseAction drops empty-string pack/template/model fields")
    func parseActionEmptyStudioPackFields() {
        let action = AgentService.parseAction(from: #"{"studio_pack_id": "", "steps": 9}"#)
        #expect(action?.studioPackId == nil)
        #expect(action?.steps == 9)
    }

    @Test("summary includes pack, template, and model when present")
    func actionSummaryIncludesPackFields() {
        var action = AgentAction()
        action.studioPackId = "life-design-healthcare"
        action.templateId = "cpr"
        action.model = "z-image-turbo"
        let s = action.summary
        #expect(s.contains("pack life-design-healthcare"))
        #expect(s.contains("template cpr"))
        #expect(s.contains("model z-image-turbo"))
    }

    @Test("hasChanges is true when only a studio pack id is set")
    func hasChangesForStudioPackOnly() {
        var action = AgentAction()
        action.studioPackId = "life-design-healthcare"
        #expect(action.hasChanges)
    }

    @Test("validationWarnings flags an unknown pack, model, and LoRA")
    func validationWarningsFlagsUnknownReferences() {
        var action = AgentAction()
        action.studioPackId = "not-a-real-pack"
        action.model = "not-a-real-model"
        action.loras = ["not-a-real-lora.safetensors"]
        let warnings = action.validationWarnings(
            availablePackIds: ["life-design-healthcare"],
            availableModelIds: ["z-image-turbo"],
            availableLoRAFilenames: ["Anneliese_Zbase3.safetensors"]
        )
        #expect(warnings.count == 3)
        #expect(warnings.contains { $0.contains("not-a-real-pack") })
        #expect(warnings.contains { $0.contains("not-a-real-model") })
        #expect(warnings.contains { $0.contains("not-a-real-lora.safetensors") })
    }

    @Test("validationWarnings is empty when everything resolves, including scaled LoRA syntax")
    func validationWarningsEmptyWhenResolved() {
        var action = AgentAction()
        action.studioPackId = "life-design-healthcare"
        action.model = "z-image-turbo"
        action.loras = ["Anneliese_Zbase3.safetensors=0.8"]
        let warnings = action.validationWarnings(
            availablePackIds: ["life-design-healthcare"],
            availableModelIds: ["z-image-turbo"],
            availableLoRAFilenames: ["Anneliese_Zbase3.safetensors"]
        )
        #expect(warnings.isEmpty)
    }
}
