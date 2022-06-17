/* Go wrapper for GPT-3. */
package gpt3

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const (
	// APIURL is the base URL for the GPT-3 API.
	APIURL = "https://api.openai.com/v1/completions"
)

// Client is a GPT-3 client.
type Client struct {
	// APIKey is the API key for the GPT-3 API.
	APIKey string

	// Logger is the logger to use.
	Logger Logger

	// HTTPClient is the HTTP client to use.
	HTTPClient *http.Client
}

// NewClient creates a new GPT-3 client.
func NewClient(apiKey string) *Client {
	return &Client{
		APIKey:     apiKey,
		HTTPClient: &http.Client{Timeout: time.Second * 10},
	}
}

// Completion is a GPT-3 completion.
type Completion struct {
	// ID is the completion ID.
	ID string `json:"id"`

	// Text is the completion text.
	Text string `json:"text"`

	// Timestamp is the completion timestamp.
	Timestamp string `json:"timestamp"`

	// Logprobs is the completion logprobs.
	Logprobs []float64 `json:"logprobs"`

	// Choices is the completion choices.
	Choices []Choice `json:"choices"`
}

// Choice is a GPT-3 completion choice.
type Choice struct {
	// Text is the choice text.
	Text string `json:"text"`

	// Logprob is the choice logprob.
	Logprob float64 `json:"logprob"`

	// Timestamp is the choice timestamp.
	Timestamp string `json:"timestamp"`
}

// Completions is a list of GPT-3 completions.
type Completions struct {
	// ID is the completion ID.
	ID string `json:"id"`

	// Completions is the list of completions.
	Completions []Completion `json:"completions"`
}

// Complete completes a prompt.
func (c *Client) Complete(prompt string, options ...Option) (*Completions, error) {
	// Create the request.
	req, err := c.createRequest(prompt, options...)
	if err != nil {
		return nil, err
	}

	// Send the request.
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// Read the response.
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	// Check the response status.
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	// Parse the response.
	var completions Completions
	if err := json.Unmarshal(body, &completions); err != nil {
		return nil, err
	}

	return &completions, nil
}

// createRequest creates a request.
func (c *Client) createRequest(prompt string, options ...Option) (*http.Request, error) {
	// Create the request.
	req, err := http.NewRequest(http.MethodPost, APIURL, nil)
	if err != nil {
		return nil, err
	}

	// Set the headers.
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.APIKey))

	// Set the query parameters.
	q := req.URL.Query()
	q.Set("prompt", prompt)
	for _, option := range options {
		option(q)
	}
	req.URL.RawQuery = q.Encode()

	return req, nil
}

// Option is a GPT-3 option.
type Option func(q url.Values)

// MaxTokens sets the maximum number of tokens to generate.
func MaxTokens(maxTokens int) Option {
	return func(q url.Values) {
		q.Set("max_tokens", fmt.Sprintf("%d", maxTokens))
	}
}

// Temperature sets the temperature.
func Temperature(temperature float64) Option {
	return func(q url.Values) {
		q.Set("temperature", fmt.Sprintf("%f", temperature))
	}
}

// TopP sets the top-p.
func TopP(topP float64) Option {
	return func(q url.Values) {
		q.Set("top_p", fmt.Sprintf("%f", topP))
	}
}

// N sets the number of completions to return.
func N(n int) Option {
	return func(q url.Values) {
		q.Set("n", fmt.Sprintf("%d", n))
	}
}

// Stream sets the stream.
func Stream(stream bool) Option {
	return func(q url.Values) {
		q.Set("stream", fmt.Sprintf("%t", stream))
	}
}

// Logprobs sets the logprobs.
func Logprobs(logprobs bool) Option {
	return func(q url.Values) {
		q.Set("logprobs", fmt.Sprintf("%t", logprobs))
	}
}

// Stop sets the stop.
func Stop(stop string) Option {
	return func(q url.Values) {
		q.Set("stop", stop)
	}
}

// Engine sets the engine.
func Engine(engine string) Option {
	return func(q url.Values) {
		q.Set("engine", engine)
	}
}

// EngineVersion sets the engine version.
func EngineVersion(engineVersion string) Option {
	return func(q url.Values) {
		q.Set("engine_version", engineVersion)
	}
}

// Presets sets the presets.
func Presets(presets ...string) Option {
	return func(q url.Values) {
		q.Set("presets", strings.Join(presets, ","))
	}
}

// Logger is a logger.
type Logger interface {
	Printf(format string, v ...interface{})
}

// LoggerFunc is a logger function.
type LoggerFunc func(format string, v ...interface{})

// Printf prints a message.
func (f LoggerFunc) Printf(format string, v ...interface{}) {
	f(format, v...)
}

// LoggerWriter is a logger writer.
type LoggerWriter struct {
	Logger Logger
}

// Write writes a message.
func (w *LoggerWriter) Write(p []byte) (n int, err error) {
	w.Logger.Printf("%s", bytes.TrimSpace(p))
	return len(p), nil
}
