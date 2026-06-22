// SSE client for the AXIOM /reason endpoint. EventSource is GET-only, so we stream the
// POST response body and parse the `event:`/`data:` frames ourselves.

export interface StepEvent {
  idx: number;
  text: string;
  per_head: Record<string, number>;
  r_aggregate: number;
  uncertainty: number;
  decision: string;
  surviving: string[];
  pruned: string[];
}

export interface DoneEvent {
  answer: string | null;
  n_steps: number;
  n_tokens: number;
}

export interface ReasonRequest {
  question: string;
  domain: string;
  answer_type: string;
}

export async function streamReason(
  body: ReasonRequest,
  onStep: (step: StepEvent) => void,
  onDone: (done: DoneEvent) => void,
): Promise<void> {
  const res = await fetch("/reason", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const contentType = res.headers.get("content-type") ?? "";
  if (!res.ok || !contentType.includes("text/event-stream")) {
    throw new Error(`reason endpoint unavailable (${res.status})`);
  }
  if (!res.body) throw new Error("no response stream");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let event = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) {
        const data = JSON.parse(line.slice(5).trim());
        if (event === "step") onStep(data as StepEvent);
        else if (event === "done") onDone(data as DoneEvent);
      }
    }
  }
}
