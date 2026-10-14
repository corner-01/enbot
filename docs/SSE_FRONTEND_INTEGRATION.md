# EnBot SSE frontend integration

## Endpoint

`POST /v1/chat/stream`

Request headers:

```http
Content-Type: application/json
Accept: text/event-stream
X-API-Key: <optional when ENBOT_API_KEY is empty>
X-Request-ID: <optional frontend trace id>
```

Request body:

```json
{
  "query": "Can the inverter support 1.5x PV oversizing?",
  "session_id": "web-user-001",
  "evaluate": false,
  "debug": false
}
```

This endpoint uses `POST`, so use `fetch()` streaming instead of the browser's
native `EventSource`, which only supports `GET` and cannot send this JSON body.

## Event contract

Events always use UTF-8 JSON in the `data:` field.

1. `meta`: request accepted; contains `request_id`, `session_id`, `model` and `protocol_version`.
2. `heartbeat`: backend is still retrieving or generating; contains elapsed seconds.
3. `delta`: append `delta` to the visible answer in ascending `index` order.
4. `sources`: replace the source list and read `retrieval_backend`.
5. `done`: successful terminal event. The same `request_id` can be sent to `/v1/feedback`.
6. `error`: failed terminal event; show `message` and stop reading.

The current backend starts sending answer chunks after the complete RAG answer is
available. Heartbeats keep the connection alive during retrieval and generation.
The event contract is ready for true model-token streaming later without frontend changes.

## Browser example

```js
export async function streamEnBot({ query, sessionId, apiKey, onDelta, onSources }) {
  const response = await fetch("http://127.0.0.1:8000/v1/chat/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      ...(apiKey ? { "X-API-Key": apiKey } : {}),
    },
    body: JSON.stringify({
      query,
      session_id: sessionId,
      evaluate: false,
      debug: false,
    }),
  });

  if (!response.ok || !response.body) {
    throw new Error(`EnBot HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let requestId = response.headers.get("X-Request-ID");

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      if (!frame.trim()) continue;

      let event = "message";
      const dataLines = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
      }
      if (!dataLines.length) continue;

      const data = JSON.parse(dataLines.join("\n"));
      requestId = data.request_id || requestId;
      if (event === "delta") onDelta(data.delta);
      if (event === "sources") onSources(data.sources, data.retrieval_backend);
      if (event === "error") throw new Error(data.message || "EnBot stream failed");
      if (event === "done") return { requestId, ...data };
    }

    if (done) break;
  }

  throw new Error("EnBot stream ended without a done event");
}
```

## Reverse proxy requirements

For Nginx, disable buffering and increase the read timeout on this route:

```nginx
location /v1/chat/stream {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 180s;
    proxy_set_header Connection "";
    proxy_set_header X-Request-ID $request_id;
}
```

Recommended frontend behavior:

- Abort the request with `AbortController` when the user presses Stop.
- Keep `session_id` stable for one conversation and unique between users.
- Do not render `data` as HTML; append it as text.
- Submit thumbs-up/down feedback with the `request_id` from `meta` or `done`.
