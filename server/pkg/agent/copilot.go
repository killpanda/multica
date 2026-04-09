package agent

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"time"
)

// copilotBackend implements Backend by spawning `copilot -p <prompt>
// --output-format=json --allow-all-tools --autopilot --no-ask-user` and reading
// streaming NDJSON events from stdout.
//
// The GitHub Copilot CLI (`@github/copilot` npm package) supports programmatic
// operation via the -p / --prompt flag and outputs JSONL events when
// --output-format=json is set.
//
// Authentication is provided via the COPILOT_GITHUB_TOKEN, GH_TOKEN, or
// GITHUB_TOKEN environment variable — set these in the daemon's agent env or
// the user's environment before starting the daemon.
type copilotBackend struct {
	cfg Config
}

func (b *copilotBackend) Execute(ctx context.Context, prompt string, opts ExecOptions) (*Session, error) {
	execPath := b.cfg.ExecutablePath
	if execPath == "" {
		execPath = "copilot"
	}
	if _, err := exec.LookPath(execPath); err != nil {
		return nil, fmt.Errorf("copilot executable not found at %q: %w", execPath, err)
	}

	timeout := opts.Timeout
	if timeout == 0 {
		timeout = 20 * time.Minute
	}
	runCtx, cancel := context.WithTimeout(ctx, timeout)

	args := []string{
		"--output-format", "json",
		"--allow-all-tools",
		"--autopilot",
		"--no-ask-user",
	}
	if opts.Model != "" {
		args = append(args, "--model", opts.Model)
	}
	if opts.SystemPrompt != "" {
		// Copilot CLI reads custom instructions from AGENTS.md in the cwd; there
		// is no direct system-prompt flag. If a system prompt is provided, prepend
		// it to the user prompt so it is still visible to the model.
		prompt = opts.SystemPrompt + "\n\n" + prompt
	}
	if opts.MaxTurns > 0 {
		args = append(args, "--max-autopilot-continues", fmt.Sprintf("%d", opts.MaxTurns))
	}
	if opts.ResumeSessionID != "" {
		args = append(args, "--resume", opts.ResumeSessionID)
	}
	args = append(args, "-p", prompt)

	cmd := exec.CommandContext(runCtx, execPath, args...)
	if opts.Cwd != "" {
		cmd.Dir = opts.Cwd
	}
	cmd.Env = buildEnv(b.cfg.Env)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("copilot stdout pipe: %w", err)
	}
	cmd.Stderr = newLogWriter(b.cfg.Logger, "[copilot:stderr] ")

	if err := cmd.Start(); err != nil {
		cancel()
		return nil, fmt.Errorf("start copilot: %w", err)
	}

	b.cfg.Logger.Info("copilot started", "pid", cmd.Process.Pid, "cwd", opts.Cwd, "model", opts.Model)

	msgCh := make(chan Message, 256)
	resCh := make(chan Result, 1)

	go func() {
		defer cancel()
		defer close(msgCh)
		defer close(resCh)

		startTime := time.Now()
		scanResult := b.processEvents(stdout, msgCh)

		exitErr := cmd.Wait()
		duration := time.Since(startTime)

		if runCtx.Err() == context.DeadlineExceeded {
			scanResult.status = "timeout"
			scanResult.errMsg = fmt.Sprintf("copilot timed out after %s", timeout)
		} else if runCtx.Err() == context.Canceled {
			scanResult.status = "aborted"
			scanResult.errMsg = "execution cancelled"
		} else if exitErr != nil && scanResult.status == "completed" {
			scanResult.status = "failed"
			scanResult.errMsg = fmt.Sprintf("copilot exited with error: %v", exitErr)
		}

		b.cfg.Logger.Info("copilot finished", "pid", cmd.Process.Pid, "status", scanResult.status, "duration", duration.Round(time.Millisecond).String())

		resCh <- Result{
			Status:     scanResult.status,
			Output:     scanResult.output,
			Error:      scanResult.errMsg,
			DurationMs: duration.Milliseconds(),
			SessionID:  scanResult.sessionID,
		}
	}()

	return &Session{Messages: msgCh, Result: resCh}, nil
}

// ── Event processing ──

type copilotScanResult struct {
	status    string
	errMsg    string
	output    string
	sessionID string
}

// processEvents reads NDJSON lines from r, dispatches events to ch, and returns
// the accumulated result.
//
// The GitHub Copilot CLI emits one JSON object per line when --output-format=json
// is used. Each event has a "type" field that identifies the kind of event.
// Common types include:
//
//   - "assistant" / "message" (role:"assistant") — text or tool-use content
//   - "tool_call" / "tool_use" — tool invocation by the agent
//   - "tool_result" — result returned from a tool
//   - "thinking" — model reasoning (if extended thinking is enabled)
//   - "result" / "done" / "session_end" — final outcome of the session
//   - "error" — an error occurred during execution
func (b *copilotBackend) processEvents(r io.Reader, ch chan<- Message) copilotScanResult {
	var output strings.Builder
	var sessionID string
	finalStatus := "completed"
	var finalError string

	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		// Attempt to parse as a JSONL event.
		var event copilotEvent
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			// Non-JSON output — treat the line as plain text from the agent.
			if line != "" {
				output.WriteString(line)
				output.WriteString("\n")
				trySend(ch, Message{Type: MessageText, Content: line})
			}
			continue
		}

		if event.SessionID != "" {
			sessionID = event.SessionID
		}

		switch event.Type {
		case "assistant", "message":
			b.handleAssistantEvent(event, ch, &output)
		case "tool_call", "tool_use":
			b.handleToolCallEvent(event, ch)
		case "tool_result", "tool_output":
			b.handleToolResultEvent(event, ch)
		case "thinking":
			if text := event.textContent(); text != "" {
				trySend(ch, Message{Type: MessageThinking, Content: text})
			}
		case "result", "done", "session_end":
			b.handleResultEvent(event, &finalStatus, &finalError, &output)
			if event.SessionID != "" {
				sessionID = event.SessionID
			}
		case "error":
			text := event.textContent()
			if text == "" {
				text = event.Error
			}
			if text != "" {
				trySend(ch, Message{Type: MessageError, Content: text})
				finalStatus = "failed"
				finalError = text
			}
		case "session_start":
			trySend(ch, Message{Type: MessageStatus, Status: "running"})
		}
	}

	return copilotScanResult{
		status:    finalStatus,
		errMsg:    finalError,
		output:    output.String(),
		sessionID: sessionID,
	}
}

func (b *copilotBackend) handleAssistantEvent(event copilotEvent, ch chan<- Message, output *strings.Builder) {
	// Handle flat text content.
	if event.Role == "" || event.Role == "assistant" {
		if text := event.textContent(); text != "" {
			output.WriteString(text)
			trySend(ch, Message{Type: MessageText, Content: text})
			return
		}
	}

	// Handle nested content array (Claude-like format):
	// {"type":"message","role":"assistant","content":[...]}
	if event.Content != nil {
		b.handleContentArray(event.Content, ch, output)
	}
}

func (b *copilotBackend) handleContentArray(raw json.RawMessage, ch chan<- Message, output *strings.Builder) {
	// Try array of content blocks.
	var blocks []copilotContentBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		// Try as a plain string.
		var text string
		if err2 := json.Unmarshal(raw, &text); err2 == nil && text != "" {
			output.WriteString(text)
			trySend(ch, Message{Type: MessageText, Content: text})
		}
		return
	}

	for _, block := range blocks {
		switch block.Type {
		case "text":
			if block.Text != "" {
				output.WriteString(block.Text)
				trySend(ch, Message{Type: MessageText, Content: block.Text})
			}
		case "thinking":
			if block.Text != "" {
				trySend(ch, Message{Type: MessageThinking, Content: block.Text})
			}
		case "tool_use", "tool_call":
			var input map[string]any
			if block.Input != nil {
				_ = json.Unmarshal(block.Input, &input)
			}
			trySend(ch, Message{
				Type:   MessageToolUse,
				Tool:   block.Name,
				CallID: block.ID,
				Input:  input,
			})
		}
	}
}

func (b *copilotBackend) handleToolCallEvent(event copilotEvent, ch chan<- Message) {
	var input map[string]any
	if event.Input != nil {
		_ = json.Unmarshal(event.Input, &input)
	}
	trySend(ch, Message{
		Type:   MessageToolUse,
		Tool:   event.toolName(),
		CallID: event.callID(),
		Input:  input,
	})
}

func (b *copilotBackend) handleToolResultEvent(event copilotEvent, ch chan<- Message) {
	trySend(ch, Message{
		Type:   MessageToolResult,
		CallID: event.toolCallID(),
		Output: event.textContent(),
	})
}

func (b *copilotBackend) handleResultEvent(event copilotEvent, status, errMsg *string, output *strings.Builder) {
	if event.Output != "" {
		output.Reset()
		output.WriteString(event.Output)
	}
	if s := event.Status; s != "" && s != "completed" && s != "success" {
		*status = "failed"
		if event.Error != "" {
			*errMsg = event.Error
		} else {
			*errMsg = fmt.Sprintf("copilot session ended with status: %s", s)
		}
	}
	if event.Error != "" && *status != "failed" {
		*status = "failed"
		*errMsg = event.Error
	}
}

// ── Copilot CLI JSONL event types ──

// copilotEvent represents a single JSONL event emitted by the Copilot CLI when
// --output-format=json is used. The schema is flexible to accommodate both flat
// and nested content formats.
//
// The "content" field is kept as json.RawMessage because the Copilot CLI may
// emit it as either a plain string or an array of content blocks depending on
// the event type.
type copilotEvent struct {
	Type      string          `json:"type"`
	Role      string          `json:"role,omitempty"`
	SessionID string          `json:"session_id,omitempty"`
	Status    string          `json:"status,omitempty"`
	Error     string          `json:"error,omitempty"`
	Output    string          `json:"output,omitempty"`

	// Content carries the primary payload. It may be a JSON string or a JSON
	// array of content blocks — use textContent() / handleContentRaw() to decode.
	Content json.RawMessage `json:"content,omitempty"`
	// Text is an alternative flat text field used by some event types.
	Text string `json:"text,omitempty"`

	// Tool call fields.
	Name       string          `json:"name,omitempty"`
	ToolName   string          `json:"tool_name,omitempty"`
	Input      json.RawMessage `json:"input,omitempty"`
	ID         string          `json:"id,omitempty"`
	CallID     string          `json:"call_id,omitempty"`
	ToolUseID  string          `json:"tool_use_id,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
}

func (e *copilotEvent) textContent() string {
	if e.Text != "" {
		return e.Text
	}
	if e.Content != nil {
		var s string
		if err := json.Unmarshal(e.Content, &s); err == nil {
			return s
		}
	}
	return ""
}

func (e *copilotEvent) toolName() string {
	if e.ToolName != "" {
		return e.ToolName
	}
	return e.Name
}

func (e *copilotEvent) callID() string {
	if e.CallID != "" {
		return e.CallID
	}
	return e.ID
}

func (e *copilotEvent) toolCallID() string {
	if e.ToolCallID != "" {
		return e.ToolCallID
	}
	return e.ToolUseID
}

// copilotContentBlock represents a single item in a content array.
type copilotContentBlock struct {
	Type  string          `json:"type"`
	Text  string          `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
}
