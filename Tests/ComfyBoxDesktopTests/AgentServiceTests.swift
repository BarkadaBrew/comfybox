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

    @Test("system prompt instructs the PROMPT: convention")
    func systemPrompt() {
        #expect(AgentService.systemPrompt.contains("PROMPT:"))
    }
}
