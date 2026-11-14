// Translators between the client wires (OpenAI Chat Completions, Anthropic Messages)
// and the three upstream wires (OpenAI, Anthropic, Gemini).
//
// The env's providers answer with an ANSWER ECHO of the request they received, so every
// translation must preserve, exactly: the system text, the user text, the tool names in
// order, max_tokens, the stop sequences, the image bytes and the JSON-mode flag. Anything
// else (temperature, top_p, tool definitions beyond names) is forwarded where the wire
// carries it but is not echoed. Usage is translated, never recomputed: the provider's
// token counts are the ones the scorer compares against.
//
// Everything here is a pure function of its arguments (or, for the SSE translators, of
// the upstream bytes): no clock, no randomness, so a replay leaves the same export.

import { isRecord, type Json } from './providers.ts';
import { sseEvent } from './wire.ts';

/** The fixed epoch second for every chat completion this server writes. */
const FIXED_CREATED = 1767225600;

/** A number, or 0 when the value is not a finite number. */
function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The text of a string or of a block array's text blocks, joined with "\n". */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** A stop value as a list of strings: a string becomes one, an array keeps its strings. */
function stopListOf(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((x): x is string => typeof x === 'string');
}

/** An image URL as base64 data (with its media type) or as a plain URL. */
function parseImageUrl(url: string): { mediaType: string; data: string } | { url: string } {
  const dataUrl = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (dataUrl !== null) return { mediaType: dataUrl[1] !== undefined && dataUrl[1] !== '' ? dataUrl[1] : 'image/png', data: dataUrl[2] };
  return { url };
}

/** One OpenAI content part as Anthropic blocks: text or an inline image. */
function openAIPartsToAnthropicBlocks(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ type: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const blocks: Json[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
    else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
      const image = parseImageUrl(part.image_url.url);
      if ('data' in image) blocks.push({ type: 'image', source: { type: 'base64', media_type: image.mediaType, data: image.data } });
      else blocks.push({ type: 'image', source: { type: 'url', url: image.url } });
    }
  }
  return blocks;
}

/** One OpenAI content part as Gemini parts: text, inline image data or a file reference. */
function openAIPartsToGeminiParts(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
    else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
      const image = parseImageUrl(part.image_url.url);
      if ('data' in image) parts.push({ inlineData: { mimeType: image.mediaType, data: image.data } });
      else parts.push({ fileData: { fileUri: image.url } });
    }
  }
  return parts;
}

/** OpenAI function tools as Anthropic tools (names, descriptions and schemas, in order). */
function openAIToolsToAnthropic(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
}

/** OpenAI function tools as Gemini function declarations (names, descriptions, parameters). */
function openAIToolsToGemini(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
}

/** Anthropic tools as Gemini function declarations. */
function anthropicToolsToGemini(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      },
    ];
  });
}

/** An OpenAI tool_choice as an Anthropic one. */
function openAIToolChoiceToAnthropic(choice: unknown): Json | null {
  if (typeof choice === 'string') {
    if (choice === 'auto') return { type: 'auto' };
    if (choice === 'required') return { type: 'any' };
    if (choice === 'none') return { type: 'none' };
    return null;
  }
  if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    return { type: 'tool', name: choice.function.name };
  }
  return null;
}

/**
 * A Chat Completions request as an Anthropic Messages request naming `model`.
 * No JSON mode is ever set: Anthropic has none, and the provider would echo
 * json_mode false, which is what an Anthropic client expects.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'user') {
      const blocks = openAIPartsToAnthropicBlocks(raw.content);
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (role === 'tool') {
      // A tool_result joins the previous user message when there is one: a lone
      // tool_result user message would overwrite the echo's user text with ''.
      const block: Json = {
        type: 'tool_result',
        tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
      };
      const last = messages.length > 0 ? messages[messages.length - 1] : undefined;
      if (last !== undefined && last.role === 'user' && Array.isArray(last.content)) (last.content as Json[]).push(block);
      else messages.push({ role: 'user', content: [block] });
      continue;
    }
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stop = stopListOf(body.stop);
  if (stop.length > 0) out.stop_sequences = stop;
  const tools = openAIToolsToAnthropic(body);
  if (tools.length > 0) out.tools = tools;
  const toolChoice = openAIToolChoiceToAnthropic(body.tool_choice);
  if (toolChoice !== null) out.tool_choice = toolChoice;
  if (body.stream === true) out.stream = true;
  return out;
}

/**
 * A Chat Completions request as a Gemini generateContent request.
 * JSON mode becomes responseMimeType application/json; without it the provider
 * echoes json_mode false, as the client expects.
 */
export function openAIToGeminiRequest(body: Json, model: string): Json {
  const contents: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'user') {
      const parts = openAIPartsToGeminiParts(raw.content);
      if (parts.length > 0) contents.push({ role: 'user', parts });
      continue;
    }
    if (role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          functionCall: {
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: argumentsOf(call.function.arguments),
          },
        });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    // A tool message is skipped: tool results do not enter the echo, and a
    // function-role content would overwrite the echo's user text with ''.
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') config.maxOutputTokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stop = stopListOf(body.stop);
  if (stop.length > 0) config.stopSequences = stop;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const tools = openAIToolsToGemini(body);
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools }];
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent request.
 * No responseMimeType is ever set: an Anthropic client expects json_mode false.
 */
export function anthropicToGeminiRequest(body: Json, model: string): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    // A message's content may be one string or an array of blocks.
    const blocks: unknown[] =
      typeof raw.content === 'string' ? [{ type: 'text', text: raw.content }] : Array.isArray(raw.content) ? raw.content : [];
    if (raw.role === 'user') {
      const parts: Json[] = [];
      for (const block of blocks) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            parts.push({
              inlineData: {
                mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png',
                data: block.source.data,
              },
            });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            parts.push({ fileData: { fileUri: block.source.url } });
          }
        }
        // tool_result blocks are skipped: they do not enter the echo.
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      for (const block of blocks) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'tool_use') {
          parts.push({
            functionCall: {
              name: typeof block.name === 'string' ? block.name : '',
              args: isRecord(block.input) ? block.input : {},
            },
          });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
  }
  const out: Json = { contents };
  const systemText = textOf(body.system);
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop_sequences)) {
    const stop = body.stop_sequences.filter((x): x is string => typeof x === 'string');
    if (stop.length > 0) config.stopSequences = stop;
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const tools = anthropicToolsToGemini(body);
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools }];
  return out;
}

/** An Anthropic stop_reason as an OpenAI finish_reason. */
function anthropicStopToOpenAI(reason: string): string {
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'max_tokens') return 'length';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as an OpenAI finish_reason. */
function geminiStopToOpenAI(reason: string): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as an Anthropic stop_reason. */
function geminiStopToAnthropic(reason: string): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** The text of a Gemini answer's first candidate, its parts joined with ''. */
function geminiTextOf(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts
    .filter(isRecord)
    .map((p) => (typeof p.text === 'string' ? p.text : ''))
    .join('');
}

/** A Gemini answer's finishReason, or null when the frame carries none. */
function geminiFinishOf(body: Json): string | null {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  return typeof first.finishReason === 'string' ? first.finishReason : null;
}

/** A Gemini answer's usage, from its usageMetadata. */
function geminiUsageOf(body: Json): { input: number; output: number } {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) };
}

/** An Anthropic Messages object as a Chat Completions object naming `model`. */
export function anthropicJsonToOpenAI(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const textParts: string[] = [];
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') textParts.push(block.text);
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
        type: 'function',
        function: {
          name: typeof block.name === 'string' ? block.name : '',
          arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
        },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = numberOrZero(usage.input_tokens);
  const output = numberOrZero(usage.output_tokens);
  const stop = typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn';
  return {
    id: `chatcmpl-${typeof body.id === 'string' ? body.id : 'sim'}`,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: textParts.join(''),
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        },
        finish_reason: anthropicStopToOpenAI(stop),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** A Gemini generateContent answer as a Chat Completions object naming `model`. */
export function geminiJsonToOpenAI(body: Json, model: string): Json {
  const usage = geminiUsageOf(body);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: geminiTextOf(body) },
        finish_reason: geminiStopToOpenAI(geminiFinishOf(body) ?? 'STOP'),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object naming `model`. */
export function geminiJsonToAnthropic(body: Json, model: string): Json {
  const usage = geminiUsageOf(body);
  return {
    id: `msg_${model}`,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: geminiTextOf(body) }],
    stop_reason: geminiStopToAnthropic(geminiFinishOf(body) ?? 'STOP'),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

// ---------------------------------------------------------------------------
// SSE translators
// ---------------------------------------------------------------------------

/** One server-sent event frame: its event name (null when unnamed) and its data. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split one raw SSE frame's text into its event name and data lines. */
function parseFrame(frame: string): SseFrame {
  let event: string | null = null;
  const data: string[] = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
    else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
  }
  return { event, data: data.join('\n') };
}

/** Read an SSE stream as frames, split on blank lines. */
export async function* readSseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        yield parseFrame(buffer.slice(0, at));
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield parseFrame(buffer);
  } finally {
    reader.releaseLock();
  }
}

/** One SSE data payload as a JSON object, or null when it is not one. */
export function parseSseJson(data: string): Json | null {
  try {
    const value: unknown = JSON.parse(data);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** One OpenAI chat chunk as an SSE data line. */
function openAIChunk(model: string, delta: Json, finish: string | null, usage?: Json): string {
  const chunk: Json = {
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  };
  if (usage !== undefined) chunk.usage = usage;
  return `data: ${JSON.stringify(chunk)}\n\n`;
}

/**
 * An Anthropic event stream as OpenAI chat chunks: a role chunk, the text and
 * tool-call deltas, a finish chunk, a usage chunk and [DONE]. A stream that ends
 * early still closes with the finish, usage and [DONE] lines.
 */
export async function* anthropicEventsToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason = 'end_turn';
  let finished = false;
  const toolArgs = new Map<number, string>();
  const toolMeta = new Map<number, { id: string; name: string }>();
  const finish = (): string[] => {
    if (finished) return [];
    finished = true;
    return [
      openAIChunk(model, {}, anthropicStopToOpenAI(stopReason)),
      openAIChunk(model, {}, null, {
        prompt_tokens: inputTokens,
        completion_tokens: outputTokens,
        total_tokens: inputTokens + outputTokens,
      }),
      'data: [DONE]\n\n',
    ];
  };
  try {
    for await (const frame of readSseFrames(upstream)) {
      const data = parseSseJson(frame.data);
      if (data === null) continue;
      if (frame.event === 'message_start') {
        if (isRecord(data.message) && isRecord(data.message.usage)) inputTokens = numberOrZero(data.message.usage.input_tokens);
        yield openAIChunk(model, { role: 'assistant', content: '' }, null);
        continue;
      }
      if (frame.event === 'content_block_start') {
        const index = typeof data.index === 'number' ? data.index : -1;
        const block = isRecord(data.content_block) ? data.content_block : {};
        if (block.type === 'tool_use' && index >= 0) {
          toolMeta.set(index, {
            id: typeof block.id === 'string' ? block.id : `toolu_${index}`,
            name: typeof block.name === 'string' ? block.name : '',
          });
          toolArgs.set(index, '');
          yield openAIChunk(model, {
            tool_calls: [{ index, id: toolMeta.get(index)?.id ?? '', type: 'function', function: { name: toolMeta.get(index)?.name ?? '', arguments: '' } }],
          }, null);
        }
        continue;
      }
      if (frame.event === 'content_block_delta') {
        const index = typeof data.index === 'number' ? data.index : -1;
        const delta = isRecord(data.delta) ? data.delta : {};
        if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
          yield openAIChunk(model, { content: delta.text }, null);
        } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '' && index >= 0) {
          toolArgs.set(index, (toolArgs.get(index) ?? '') + delta.partial_json);
          const meta = toolMeta.get(index);
          yield openAIChunk(model, {
            tool_calls: [{ index, id: meta?.id ?? '', type: 'function', function: { name: meta?.name ?? '', arguments: delta.partial_json } }],
          }, null);
        }
        continue;
      }
      if (frame.event === 'message_delta') {
        if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stopReason = data.delta.stop_reason;
        if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
        continue;
      }
      if (frame.event === 'message_stop') {
        for (const event of finish()) yield event;
      }
    }
  } finally {
    for (const event of finish()) yield event;
  }
}

/**
 * A Gemini streamGenerateContent stream as OpenAI chat chunks: a role chunk, the
 * text deltas, a finish chunk, a usage chunk (from the final frame's
 * usageMetadata) and [DONE].
 */
export async function* geminiEventsToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let inputTokens = 0;
  let outputTokens = 0;
  let finishReason = 'STOP';
  let finished = false;
  const finish = (): string[] => {
    if (finished) return [];
    finished = true;
    return [
      openAIChunk(model, {}, geminiStopToOpenAI(finishReason)),
      openAIChunk(model, {}, null, {
        prompt_tokens: inputTokens,
        completion_tokens: outputTokens,
        total_tokens: inputTokens + outputTokens,
      }),
      'data: [DONE]\n\n',
    ];
  };
  try {
    yield openAIChunk(model, { role: 'assistant', content: '' }, null);
    for await (const frame of readSseFrames(upstream)) {
      const body = parseSseJson(frame.data);
      if (body === null) continue;
      const text = geminiTextOf(body);
      if (text !== '') yield openAIChunk(model, { content: text }, null);
      const fr = geminiFinishOf(body);
      if (fr !== null) finishReason = fr;
      const usage = geminiUsageOf(body);
      if (usage.input !== 0 || usage.output !== 0) {
        inputTokens = usage.input;
        outputTokens = usage.output;
      }
    }
  } finally {
    for (const event of finish()) yield event;
  }
}

/**
 * A Gemini streamGenerateContent stream as Anthropic events: message_start, one
 * text block, message_delta with the stop reason and output tokens, message_stop.
 * The input count is not known until the final frame, so message_start carries 0;
 * the scorer compares only the output count on an Anthropic stream.
 */
export async function* geminiEventsToAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let outputTokens = 0;
  let finishReason = 'STOP';
  let finished = false;
  const finish = (): string[] => {
    if (finished) return [];
    finished = true;
    return [
      sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
      sseEvent('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: geminiStopToAnthropic(finishReason), stop_sequence: null },
        usage: { output_tokens: outputTokens },
      }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };
  try {
    yield sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 0, output_tokens: 0 },
      },
    });
    yield sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
    for await (const frame of readSseFrames(upstream)) {
      const body = parseSseJson(frame.data);
      if (body === null) continue;
      const text = geminiTextOf(body);
      if (text !== '') {
        yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } });
      }
      const fr = geminiFinishOf(body);
      if (fr !== null) finishReason = fr;
      const usage = geminiUsageOf(body);
      if (usage.output !== 0) outputTokens = usage.output;
    }
  } finally {
    for (const event of finish()) yield event;
  }
}
