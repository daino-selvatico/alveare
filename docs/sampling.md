# 🎛️ Sampling Controls & API Reference

Alveare provides fine-grained control over text generation and sampling parameters. Settings can be configured visually through the **React Web UI Dashboard** or passed directly into the **OpenAI-compatible REST API** (`/v1/chat/completions`).

---

## ⚙️ Generation Settings & Parameters

| Parameter | Type | Default | Range | Description |
|---|---|---|---|---|
| `temperature` | `float` | `1.0` | `0.0` - `2.0` | Controls randomness. `0.0` enables deterministic greedy decoding (`argmax`). Higher values increase output diversity. |
| `top_p` | `float` | `0.9` | `0.0` - `1.0` | Nucleus sampling. Filters candidate tokens to the smallest set whose cumulative probability reaches `top_p`. |
| `top_k` | `int` | `50` | `0` - `500` | Restricts next-token selection to the `k` most probable candidate tokens. Set `0` to disable top-k filtering. |
| `seed` | `int` / `null` | `null` | Integer | Optional seed for reproducible pseudo-random sampling. |
| `max_tokens` | `int` | `512` | `1` - `4096` | Maximum number of new tokens generated in the completion. |
| `max_context_length` | `int` | `4096` | `512` - `128000` | Context window constraint used for prompt truncation and KV cache budgeting. |
| `enable_thinking` | `boolean` | `true` | `true` / `false` | Enables internal reasoning / thinking tags for models that support reasoning output blocks (e.g. Gemma 4). |
| `system` | `string` / `message` | `null` | Text | System prompt instruction supplied to condition model behavior. |

---

## 🖥️ Web UI Generation Settings Panel

In the React Web UI dashboard (`http://127.0.0.1:8000`):

1. Click the **"Settings" / ⚙️** icon in the chat playground header or side drawer.
2. Adjust sliders for **Temperature**, **Top-P**, **Top-K**, and **Max Tokens**.
3. Toggle **Enable Thinking** on or off.
4. Set a default **System Prompt** for new chat conversations.
5. All preferences are automatically persisted in browser `localStorage`.

---

## 🔌 OpenAI-Compatible API (`/v1/chat/completions`)

Alveare exposes an HTTP POST endpoint at `/v1/chat/completions` that matches the OpenAI API spec.

### Request Body Schema (JSON)

```json
{
  "model": "gemma4",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful software engineer assistant."
    },
    {
      "role": "user",
      "content": "Write a Python function to check if a string is a palindrome."
    }
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "max_tokens": 256,
  "max_context_length": 4096,
  "enable_thinking": true,
  "stream": true
}
```

---

## 💻 Integration Code Examples

### 1. Python (Official `openai` Library)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="gemma4",
    messages=[
        {"role": "system", "content": "Be concise and direct."},
        {"role": "user", "content": "Explain quantum computing in one paragraph."}
    ],
    temperature=0.7,
    top_p=0.9,
    max_tokens=150,
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)
print()
```

### 2. cURL (Streaming SSE)

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4-e4b",
    "messages": [
      {"role": "user", "content": "List 3 advantages of NPU hardware acceleration."}
    ],
    "temperature": 0.5,
    "top_k": 30,
    "max_tokens": 128,
    "stream": true
  }'
```

### 3. JavaScript / Fetch API

```javascript
const response = await fetch('http://127.0.0.1:8000/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'gemma3',
    messages: [{ role: 'user', content: 'Ciao!' }],
    temperature: 0.8,
    max_tokens: 64
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);
```
