package agent

import (
	"log/slog"
	"strings"
	"testing"
)

func TestNewReturnsCopilotBackend(t *testing.T) {
	t.Parallel()
	b, err := New("copilot", Config{ExecutablePath: "/nonexistent/copilot"})
	if err != nil {
		t.Fatalf("New(copilot) error: %v", err)
	}
	if _, ok := b.(*copilotBackend); !ok {
		t.Fatalf("expected *copilotBackend, got %T", b)
	}
}

// ── textContent helper tests ──

func TestCopilotEventTextContentFromText(t *testing.T) {
	t.Parallel()
	e := copilotEvent{Text: "hello from text"}
	if got := e.textContent(); got != "hello from text" {
		t.Errorf("got %q, want %q", got, "hello from text")
	}
}

func TestCopilotEventTextContentFromContentString(t *testing.T) {
	t.Parallel()
	e := copilotEvent{}
	e.Content = []byte(`"hello from content"`)
	if got := e.textContent(); got != "hello from content" {
		t.Errorf("got %q, want %q", got, "hello from content")
	}
}

func TestCopilotEventTextContentTextTakesPrecedence(t *testing.T) {
	t.Parallel()
	e := copilotEvent{Text: "text wins"}
	e.Content = []byte(`"content ignored"`)
	if got := e.textContent(); got != "text wins" {
		t.Errorf("got %q, want %q", got, "text wins")
	}
}

func TestCopilotEventTextContentEmpty(t *testing.T) {
	t.Parallel()
	e := copilotEvent{}
	if got := e.textContent(); got != "" {
		t.Errorf("expected empty, got %q", got)
	}
}

// ── toolName / callID / toolCallID helper tests ──

func TestCopilotEventToolNamePreferToolName(t *testing.T) {
	t.Parallel()
	e := copilotEvent{ToolName: "shell", Name: "bash"}
	if got := e.toolName(); got != "shell" {
		t.Errorf("got %q, want %q", got, "shell")
	}
}

func TestCopilotEventToolNameFallbackToName(t *testing.T) {
	t.Parallel()
	e := copilotEvent{Name: "view"}
	if got := e.toolName(); got != "view" {
		t.Errorf("got %q, want %q", got, "view")
	}
}

func TestCopilotEventCallIDPreferCallID(t *testing.T) {
	t.Parallel()
	e := copilotEvent{CallID: "c1", ID: "i1"}
	if got := e.callID(); got != "c1" {
		t.Errorf("got %q, want %q", got, "c1")
	}
}

func TestCopilotEventToolCallIDPreferToolCallID(t *testing.T) {
	t.Parallel()
	e := copilotEvent{ToolCallID: "tc1", ToolUseID: "tu1"}
	if got := e.toolCallID(); got != "tc1" {
		t.Errorf("got %q, want %q", got, "tc1")
	}
}

// ── processEvents integration tests ──

func TestCopilotProcessEventsFlatTextEvent(t *testing.T) {
	t.Parallel()

	b := &copilotBackend{cfg: Config{Logger: slog.Default()}}
	ch := make(chan Message, 256)

	lines := strings.Join([]string{
		`{"type":"session_start","session_id":"ses_abc"}`,
		`{"type":"assistant","role":"assistant","text":"Hello from copilot"}`,
		`{"type":"result","status":"completed","output":"Hello from copilot","session_id":"ses_abc"}`,
	}, "\n")

	result := b.processEvents(strings.NewReader(lines), ch)

	if result.status != "completed" {
		t.Errorf("status: got %q, want %q", result.status, "completed")
	}
	if result.sessionID != "ses_abc" {
		t.Errorf("sessionID: got %q, want %q", result.sessionID, "ses_abc")
	}
	if result.errMsg != "" {
		t.Errorf("errMsg: got %q, want empty", result.errMsg)
	}

	close(ch)
	var msgs []Message
	for m := range ch {
		msgs = append(msgs, m)
	}

	// Should have: status(running), text
	var statusMsgs, textMsgs int
	for _, m := range msgs {
		switch m.Type {
		case MessageStatus:
			statusMsgs++
		case MessageText:
			textMsgs++
			if m.Content != "Hello from copilot" {
				t.Errorf("text content: got %q, want %q", m.Content, "Hello from copilot")
			}
		}
	}
	if statusMsgs != 1 {
		t.Errorf("expected 1 status message, got %d", statusMsgs)
	}
	if textMsgs != 1 {
		t.Errorf("expected 1 text message, got %d", textMsgs)
	}
}

func TestCopilotProcessEventsNestedContentArray(t *testing.T) {
	t.Parallel()

	b := &copilotBackend{cfg: Config{Logger: slog.Default()}}
	ch := make(chan Message, 256)

	// Claude-like nested content format.
	lines := `{"type":"message","role":"assistant","content":[{"type":"text","text":"Nested text"},{"type":"tool_use","id":"call_1","name":"view","input":{"path":"main.go"}}]}`

	result := b.processEvents(strings.NewReader(lines), ch)

	if result.status != "completed" {
		t.Errorf("status: got %q, want %q", result.status, "completed")
	}
	if result.output != "Nested text" {
		t.Errorf("output: got %q, want %q", result.output, "Nested text")
	}

	close(ch)
	var msgs []Message
	for m := range ch {
		msgs = append(msgs, m)
	}

	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages (text + tool_use), got %d: %+v", len(msgs), msgs)
	}
	if msgs[0].Type != MessageText || msgs[0].Content != "Nested text" {
		t.Errorf("msg[0]: got %+v", msgs[0])
	}
	if msgs[1].Type != MessageToolUse || msgs[1].Tool != "view" || msgs[1].CallID != "call_1" {
		t.Errorf("msg[1]: got %+v", msgs[1])
	}
}

func TestCopilotProcessEventsToolCallAndResult(t *testing.T) {
	t.Parallel()

	b := &copilotBackend{cfg: Config{Logger: slog.Default()}}
	ch := make(chan Message, 256)

	lines := strings.Join([]string{
		`{"type":"tool_call","name":"bash","id":"call_bash","input":{"command":"pwd"}}`,
		`{"type":"tool_result","tool_call_id":"call_bash","text":"/workspace\n"}`,
	}, "\n")

	result := b.processEvents(strings.NewReader(lines), ch)

	if result.status != "completed" {
		t.Errorf("status: got %q", result.status)
	}

	close(ch)
	var msgs []Message
	for m := range ch {
		msgs = append(msgs, m)
	}

	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d: %+v", len(msgs), msgs)
	}
	if msgs[0].Type != MessageToolUse || msgs[0].Tool != "bash" || msgs[0].CallID != "call_bash" {
		t.Errorf("msg[0]: got %+v", msgs[0])
	}
	if msgs[1].Type != MessageToolResult || msgs[1].CallID != "call_bash" || msgs[1].Output != "/workspace\n" {
		t.Errorf("msg[1]: got %+v", msgs[1])
	}
}

func TestCopilotProcessEventsThinkingEvent(t *testing.T) {
	t.Parallel()

	b := &copilotBackend{cfg: Config{Logger: slog.Default()}}
	ch := make(chan Message, 256)

	lines := `{"type":"thinking","text":"Let me reason through this..."}`

	result := b.processEvents(strings.NewReader(lines), ch)

	if result.status != "completed" {
		t.Errorf("status: got %q", result.status)
	}

	close(ch)
	var msgs []Message
	for m := range ch {
		msgs = append(msgs, m)
	}

	if len(msgs) != 1 || msgs[0].Type != MessageThinking {
		t.Fatalf("expected 1 thinking message, got %d: %+v", len(msgs), msgs)
	}
	if msgs[0].Content != "Let me reason through this..." {
		t.Errorf("content: got %q", msgs[0].Content)
	}
}

func TestCopilotProcessEventsErrorEvent(t *testing.T) {
	t.Parallel()

	b := &copilotBackend{cfg: Config{Logger: slog.Default()}}
	ch := make(chan Message, 256)

	lines := `{"type":"error","text":"Authentication failed"}`

	result := b.processEvents(strings.NewReader(lines), ch)

	if result.status != "failed" {
		t.Errorf("status: got %q, want %q", result.status, "failed")
	}
	if result.errMsg != "Authentication failed" {
		t.Errorf("errMsg: got %q, want %q", result.errMsg, "Authentication failed")
	}

	close(ch)
	var msgs []Message
	for m := range ch {
		msgs = append(msgs, m)
	}

	var errorMsgs int
	for _, m := range msgs {
		if m.Type == MessageError {
			errorMsgs++
		}
	}
	if errorMsgs != 1 {
		t.Errorf("expected 1 error message, got %d", errorMsgs)
	}
}

func TestCopilotProcessEventsResultFailedStatus(t *testing.T) {
	t.Parallel()

	b := &copilotBackend{cfg: Config{Logger: slog.Default()}}
	ch := make(chan Message, 256)

	lines := `{"type":"result","status":"error","error":"model overloaded","session_id":"ses_fail"}`

	result := b.processEvents(strings.NewReader(lines), ch)

	if result.status != "failed" {
		t.Errorf("status: got %q, want %q", result.status, "failed")
	}
	if result.errMsg != "model overloaded" {
		t.Errorf("errMsg: got %q, want %q", result.errMsg, "model overloaded")
	}
	if result.sessionID != "ses_fail" {
		t.Errorf("sessionID: got %q, want %q", result.sessionID, "ses_fail")
	}

	close(ch)
}

func TestCopilotProcessEventsNonJSONFallback(t *testing.T) {
	t.Parallel()

	b := &copilotBackend{cfg: Config{Logger: slog.Default()}}
	ch := make(chan Message, 256)

	// Non-JSON output should be treated as plain text.
	lines := "This is plain text output from copilot"

	result := b.processEvents(strings.NewReader(lines), ch)

	if result.status != "completed" {
		t.Errorf("status: got %q", result.status)
	}
	if !strings.Contains(result.output, "This is plain text output from copilot") {
		t.Errorf("output: got %q, want it to contain the plain text", result.output)
	}

	close(ch)
	var msgs []Message
	for m := range ch {
		msgs = append(msgs, m)
	}

	if len(msgs) != 1 || msgs[0].Type != MessageText {
		t.Fatalf("expected 1 text message, got %d: %+v", len(msgs), msgs)
	}
}

func TestCopilotProcessEventsSessionIDFromAnyEvent(t *testing.T) {
	t.Parallel()

	b := &copilotBackend{cfg: Config{Logger: slog.Default()}}
	ch := make(chan Message, 256)

	lines := strings.Join([]string{
		`{"type":"session_start","session_id":"ses_first"}`,
		`{"type":"assistant","text":"hi","session_id":"ses_updated"}`,
	}, "\n")

	result := b.processEvents(strings.NewReader(lines), ch)

	if result.sessionID != "ses_updated" {
		t.Errorf("sessionID: got %q, want %q (should use last seen)", result.sessionID, "ses_updated")
	}

	close(ch)
}

func TestCopilotProcessEventsEmptyAndBlankLines(t *testing.T) {
	t.Parallel()

	b := &copilotBackend{cfg: Config{Logger: slog.Default()}}
	ch := make(chan Message, 256)

	lines := strings.Join([]string{
		"",
		"   ",
		`{"type":"assistant","text":"valid"}`,
		"",
	}, "\n")

	result := b.processEvents(strings.NewReader(lines), ch)

	if result.status != "completed" {
		t.Errorf("status: got %q", result.status)
	}
	if result.output != "valid" {
		t.Errorf("output: got %q, want %q", result.output, "valid")
	}

	close(ch)
}
