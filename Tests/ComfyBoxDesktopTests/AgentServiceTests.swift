// AgentServiceTests.swift — Agent request assembly + response parsing

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("AgentService")
struct AgentServiceTests {
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
}
