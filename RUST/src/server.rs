// Server: HTTP + OpenAI-compatible API + built-in web UI, up to 4 models at once.
//   POST /v1/chat/completions  (streaming and non-streaming)
//   GET  /v1/models
//   GET  /                     chat page
// Modes: "model":"<name>" | "mode":"parallel" (every model answers) | "mode":"chain" (each model
// refines the previous answer) | "mode":"loop" (the chain repeated "rounds" times, also with one model)

use crate::gguf::Gguf;
use crate::json::{quote, Json};
use crate::model::Model;
use crate::sampler::Sampler;
use crate::tokenizer::{Msg, Tokenizer, Utf8Out};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::time::{Duration, Instant};

pub const MAX_MODELS: usize = 4;
const MAX_STAGES: usize = MAX_MODELS * 8; // limit on chain/loop stages
const MAX_MSGS: usize = 64;
const REQ_LIMIT: usize = 1 << 20; // 1 MB

/// refinement message between chain stages
const CHAIN_REFINE_MSG: &str = "The assistant message above is a draft answer to my request. \
Improve it: fix any mistakes, fill in anything missing, and reply with only the final answer.";

/// one model: vocabulary + weights + its own KV cache
struct Engine {
    tk: Tokenizer,
    m: Model,
    name: String,
}

enum RunErr {
    TooLong,      // the prompt does not fit in the context
    Failed,       // forward failed
    Disconnected, // the client went away while streaming
}

struct RunOut {
    text: String,
    prompt_toks: usize,
    gen_toks: usize,
}

fn send(s: &mut TcpStream, data: &[u8]) -> bool {
    s.write_all(data).is_ok()
}

fn http_reply(s: &mut TcpStream, code: u16, ctype: &str, body: &[u8]) {
    let st = match code {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        413 => "Payload Too Large",
        _ => "Error",
    };
    let hdr = format!(
        "HTTP/1.1 {code} {st}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    if send(s, hdr.as_bytes()) && !body.is_empty() {
        send(s, body);
    }
}

fn json_err(s: &mut TcpStream, code: u16, msg: &str) {
    http_reply(s, code, "application/json", format!("{{\"error\":{}}}", quote(msg)).as_bytes());
}

/// one SSE chunk with a piece of the reply
fn sse_chunk(model: &str, text: &str) -> String {
    format!(
        "data: {{\"id\":\"chatcmpl-fg\",\"object\":\"chat.completion.chunk\",\"model\":{},\
         \"choices\":[{{\"index\":0,\"delta\":{{\"content\":{}}},\"finish_reason\":null}}]}}\n\n",
        quote(model),
        quote(text)
    )
}

/// Runs one engine on a conversation: builds the prompt (dropping the oldest turns when the
/// context overflows), reuses the cached prefix, generates. With `stream` the tokens go out as
/// SSE chunks, preceded by a section header when there is a label (parallel/chain).
fn run_engine(
    e: &mut Engine,
    msgs: &[Msg],
    max_tokens: usize,
    mut stream: Option<&mut TcpStream>,
    label: Option<&str>,
    sp: &mut Sampler,
) -> Result<RunOut, RunErr> {
    let n_ctx = e.m.n_ctx;
    let mut max_tokens = max_tokens.max(1);

    // build the prompt — on overflow drop the oldest turns (keep the first system message and
    // the last message, and leave room for the reply)
    let keep_sys = msgs[0].role == "system";
    let mut first = keep_sys as usize;
    let mut dropped = 0;
    let reserve = max_tokens.min(256);
    let ids = loop {
        let mut tmp: Vec<Msg> = Vec::with_capacity(msgs.len());
        if keep_sys {
            tmp.push(msgs[0].clone());
        }
        tmp.extend(msgs[first..].iter().cloned());
        let ids = e.tk.encode_chat(&tmp, e.tk.add_bos, false);
        if ids.len() + reserve <= n_ctx - 8 {
            break ids;
        }
        if first < msgs.len() - 1 {
            let step = if msgs.len() - 1 - first >= 2 { 2 } else { 1 };
            first += step;
            dropped += step;
            continue;
        }
        if ids.len() < n_ctx - 8 {
            break ids;
        }
        return Err(RunErr::TooLong);
    };
    if dropped > 0 {
        println!("[{}] context overflow: dropped {dropped} oldest msgs", e.name);
    }
    max_tokens = max_tokens.min(n_ctx - ids.len());

    // cache reuse: the longest common prefix with the previous request
    let keep = e.m.reusable_prefix(&ids);
    println!(
        "[{}] {} msgs, {} prompt tokens ({keep} cached), max_gen={max_tokens}, stream={}",
        e.name,
        msgs.len(),
        ids.len(),
        stream.is_some() as i32
    );

    // section header (parallel/chain) — sent at once so the user sees who is answering
    if let (Some(s), Some(label)) = (stream.as_deref_mut(), label) {
        if !send(s, sse_chunk(&e.name, &format!("\n### {label}\n")).as_bytes()) {
            return Err(RunErr::Disconnected);
        }
    }

    // read the prompt from the point of divergence, in batches
    let t0 = Instant::now();
    e.m.seq_ids.truncate(keep);
    if let Err(err) = e.m.forward_batch(&ids[keep..], keep) {
        println!("[{}] forward failed: {err}", e.name);
        return Err(RunErr::Failed);
    }
    e.m.seq_ids.extend_from_slice(&ids[keep..]);
    let t_prompt = t0.elapsed().as_secs_f64();

    let mut text: Vec<u8> = Vec::new();
    let mut u8out = Utf8Out::new();
    let mut piece = Vec::new();
    let mut n_gen = 0;
    let mut pos = ids.len();
    let t1 = Instant::now();
    for _ in 0..max_tokens {
        let next = sp.sample(&e.tk, &mut e.m.logits, &e.m.seq_ids);
        if e.tk.is_stop(next) {
            break;
        }
        let bytes = e.tk.token_bytes(next);
        text.extend_from_slice(&bytes);
        if let Some(s) = stream.as_deref_mut() {
            // whole UTF-8 characters only: a token may end in the middle of one
            piece.clear();
            u8out.push(&bytes, &mut piece);
            if !piece.is_empty() && !send(s, sse_chunk(&e.name, &String::from_utf8_lossy(&piece)).as_bytes()) {
                return Err(RunErr::Disconnected);
            }
        }
        n_gen += 1;
        if e.m.forward(next, pos).is_err() {
            break;
        }
        e.m.seq_ids.push(next);
        pos += 1;
    }
    if let Some(s) = stream.as_deref_mut() {
        piece.clear();
        u8out.flush(&mut piece);
        if !piece.is_empty() && !send(s, sse_chunk(&e.name, &String::from_utf8_lossy(&piece)).as_bytes()) {
            return Err(RunErr::Disconnected);
        }
    }
    let t_gen = t1.elapsed().as_secs_f64();
    println!(
        "[{}] done: prompt {:.1}s ({:.1} t/s, {keep} reused) | gen {n_gen} tok @ {:.2} t/s",
        e.name,
        t_prompt,
        (ids.len() - keep) as f64 / if t_prompt > 0.0 { t_prompt } else { 1.0 },
        n_gen as f64 / if t_gen > 0.0 { t_gen } else { 1.0 }
    );
    Ok(RunOut { text: String::from_utf8_lossy(&text).into_owned(), prompt_toks: ids.len(), gen_toks: n_gen })
}

/// the "messages" array; content may be a string or a list of {type:"text", text:...} parts
fn parse_messages(body: &Json) -> Option<Vec<Msg>> {
    let Some(Json::Arr(arr)) = body.get("messages") else { return None };
    let mut out = Vec::new();
    for m in arr.iter().take(MAX_MSGS) {
        let Some(role) = m.str("role") else { continue };
        let content = match m.get("content") {
            Some(Json::Str(s)) => s.clone(),
            Some(Json::Arr(parts)) => parts.iter().filter_map(|p| p.str("text")).collect::<Vec<_>>().join(""),
            Some(_) => String::new(),
            None => continue,
        };
        out.push(Msg { role: role.chars().take(15).collect(), content });
    }
    Some(out)
}

fn is_blank(s: &str) -> bool {
    s.trim().is_empty()
}

fn handle_chat(s: &mut TcpStream, engs: &mut [Engine], body: &str) {
    let body = match Json::parse(body) {
        Ok(j) => j,
        Err(e) => return json_err(s, 400, &format!("invalid JSON: {e}")),
    };
    let msgs = match parse_messages(&body) {
        Some(m) if !m.is_empty() => m,
        _ => return json_err(s, 400, "missing or invalid messages"),
    };
    let max_tokens = body.num("max_tokens", 1024.0).max(1.0) as usize;
    let stream = body.bool("stream", false);

    // sampling: temperature <= 0 (the default) is greedy
    let mut smp = Sampler::default();
    smp.temp = (body.num("temperature", 0.0) as f32).max(0.0);
    smp.top_p = body.num("top_p", 0.95) as f32;
    smp.top_k = body.num("top_k", 40.0) as i32;
    smp.repeat_penalty = body.num("repeat_penalty", 1.1) as f32;
    smp.repeat_last_n = body.num("repeat_last_n", 64.0).max(0.0) as usize;
    let seed = body.num("seed", -1.0);
    smp.rng = if seed >= 0.0 {
        seed as u64
    } else {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)
    };
    if smp.rng == 0 {
        smp.rng = 0x9E37_79B9_7F4A_7C15;
    }

    let mut want = body.str("model").unwrap_or("").to_string();
    let mut mode = body.str("mode").unwrap_or("").to_string();
    // the mode through the model field too — for OpenAI clients that cannot send extra fields
    if mode.is_empty() && matches!(want.as_str(), "parallel" | "chain" | "layers" | "loop") {
        mode = std::mem::take(&mut want);
    }
    let n_eng = engs.len();
    let m_par = mode == "parallel" && n_eng > 1;
    let m_chn = matches!(mode.as_str(), "chain" | "layers" | "pipeline") && n_eng > 1;
    let m_loop = mode == "loop";
    let rounds = (body.num("rounds", 2.0) as i64).clamp(1, 8) as usize;

    // single mode: match the name, otherwise the first model (OpenAI client compatibility)
    let sel = if want.is_empty() {
        0
    } else {
        engs.iter().position(|e| e.name == want || e.name.contains(&want)).unwrap_or(0)
    };

    if stream {
        let hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\
                   Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
        if !send(s, hdr.as_bytes()) {
            return;
        }
    }

    let mut texts: Vec<(String, String)> = Vec::new(); // (model, text)
    let (mut sum_p, mut sum_g) = (0usize, 0usize);
    let resp_model: String;

    if m_par {
        // every model answers the same question — one after the other (each uses the whole CPU)
        resp_model = "parallel".into();
        for e in engs.iter_mut() {
            let label = e.name.clone();
            let st = stream.then_some(&mut *s);
            match run_engine(e, &msgs, max_tokens, st, Some(&label), &mut smp) {
                Ok(o) => {
                    sum_p += o.prompt_toks;
                    sum_g += o.gen_toks;
                    texts.push((label, o.text));
                }
                Err(RunErr::Disconnected) => return,
                Err(_) => texts.push((label, "[error: prompt too long or generation failed]".into())),
            }
        }
    } else if m_chn || m_loop {
        // layers: a draft, then refinement one model after another; loop repeats the chain
        resp_model = if m_loop { "loop" } else { "chain" }.into();
        let n_stages = if m_loop { n_eng * rounds } else { n_eng }.min(MAX_STAGES);
        let mut draft: Option<String> = None;
        for i in 0..n_stages {
            let e = &mut engs[i % n_eng];
            let mut cm = msgs.clone();
            if let Some(d) = &draft {
                cm.push(Msg { role: "assistant".into(), content: d.clone() });
                cm.push(Msg { role: "user".into(), content: CHAIN_REFINE_MSG.into() });
            }
            let label = format!(
                "{}{}",
                e.name,
                if i == n_stages - 1 {
                    " (final)"
                } else if i > 0 {
                    " (refine)"
                } else {
                    " (draft)"
                }
            );
            let name = e.name.clone();
            let st = stream.then_some(&mut *s);
            match run_engine(e, &cm, max_tokens, st, Some(&label), &mut smp) {
                Ok(o) => {
                    sum_p += o.prompt_toks;
                    sum_g += o.gen_toks;
                    // an empty draft does not replace the previous one
                    if !is_blank(&o.text) {
                        draft = Some(o.text.clone());
                    }
                    texts.push((name, o.text));
                }
                Err(RunErr::Disconnected) => return,
                Err(_) => break, // keep the last good draft
            }
        }
        if texts.is_empty() {
            if !stream {
                json_err(s, 500, "generation failed");
            }
            return;
        }
    } else {
        let e = &mut engs[sel];
        resp_model = e.name.clone();
        let st = stream.then_some(&mut *s);
        match run_engine(e, &msgs, max_tokens, st, None, &mut smp) {
            Ok(o) => {
                sum_p = o.prompt_toks;
                sum_g = o.gen_toks;
                texts.push((resp_model.clone(), o.text));
            }
            Err(err) => {
                if !stream {
                    match err {
                        RunErr::TooLong => json_err(s, 400, "prompt too long for context"),
                        _ => json_err(s, 500, "forward failed"),
                    }
                }
                return;
            }
        }
    }

    if stream {
        let done = "data: {\"id\":\"chatcmpl-fg\",\"object\":\"chat.completion.chunk\",\
                    \"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
        send(s, done.as_bytes());
        return;
    }
    let usage = format!(
        "\"usage\":{{\"prompt_tokens\":{sum_p},\"completion_tokens\":{sum_g},\"total_tokens\":{}}}",
        sum_p + sum_g
    );
    let resp = if m_par {
        let choices: Vec<String> = texts
            .iter()
            .enumerate()
            .map(|(i, (m, t))| {
                format!(
                    "{{\"index\":{i},\"model\":{},\"message\":{{\"role\":\"assistant\",\"content\":{}}},\"finish_reason\":\"stop\"}}",
                    quote(m),
                    quote(t)
                )
            })
            .collect();
        format!(
            "{{\"id\":\"chatcmpl-fg\",\"object\":\"chat.completion\",\"model\":\"parallel\",\"choices\":[{}],{usage}}}",
            choices.join(",")
        )
    } else if m_chn || m_loop {
        let stages: Vec<String> =
            texts.iter().map(|(m, t)| format!("{{\"model\":{},\"content\":{}}}", quote(m), quote(t))).collect();
        // the final answer = the last non-empty stage (a weak model may return nothing)
        let fin = texts.iter().rposition(|(_, t)| !is_blank(t)).unwrap_or(texts.len() - 1);
        format!(
            "{{\"id\":\"chatcmpl-fg\",\"object\":\"chat.completion\",\"model\":{},\"stages\":[{}],\
             \"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":{}}},\"finish_reason\":\"stop\"}}],{usage}}}",
            quote(&resp_model),
            stages.join(","),
            quote(&texts[fin].1)
        )
    } else {
        format!(
            "{{\"id\":\"chatcmpl-fg\",\"object\":\"chat.completion\",\"model\":{},\
             \"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":{}}},\"finish_reason\":\"stop\"}}],{usage}}}",
            quote(&resp_model),
            quote(&texts[0].1)
        )
    };
    http_reply(s, 200, "application/json", resp.as_bytes());
}

struct Request {
    method: String,
    url: String,
    headers: Vec<(String, String)>, // names in lower case
    body: Vec<u8>,
}

/// reads the headers, then the body by Content-Length
fn read_request(s: &mut TcpStream) -> Result<Request, u16> {
    let mut buf = Vec::with_capacity(8192);
    let mut tmp = [0u8; 16384];
    let hdr_end = loop {
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p;
        }
        if buf.len() > 64 * 1024 {
            return Err(413);
        }
        match s.read(&mut tmp) {
            Ok(0) | Err(_) => return Err(0),
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
        }
    };
    let head = String::from_utf8_lossy(&buf[..hdr_end]).into_owned();
    let mut lines = head.split("\r\n");
    let mut rl = lines.next().unwrap_or("").split_whitespace();
    let method = rl.next().unwrap_or("").to_string();
    let url = rl.next().unwrap_or("").to_string();
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    let clen: usize = headers.iter().find(|(k, _)| k == "content-length").and_then(|(_, v)| v.parse().ok()).unwrap_or(0);
    if clen > REQ_LIMIT {
        return Err(413);
    }
    let mut body = buf[hdr_end + 4..].to_vec();
    while body.len() < clen {
        match s.read(&mut tmp) {
            Ok(0) | Err(_) => break,
            Ok(n) => body.extend_from_slice(&tmp[..n]),
        }
    }
    body.truncate(clen);
    Ok(Request { method, url, headers, body })
}

/// model name = file name without the folder and without .gguf
fn model_name(p: &std::path::Path) -> String {
    let n = p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
    n.strip_suffix(".gguf").map(str::to_string).unwrap_or(n)
}

pub fn cmd_serve(paths: &[PathBuf], port: u16, n_ctx: usize) -> Result<(), String> {
    let mut engs: Vec<Engine> = Vec::new();
    for (i, p) in paths.iter().take(MAX_MODELS).enumerate() {
        println!("loading model {}/{}: {}", i + 1, paths.len().min(MAX_MODELS), p.display());
        let g = Gguf::open(p)?;
        let tk = Tokenizer::load(&g)?;
        let m = Model::load(g, n_ctx)?;
        let mut name = model_name(p);
        if engs.iter().any(|e| e.name == name) {
            name = format!("{name}-{}", i + 1); // duplicate names get a number
        }
        engs.push(Engine { tk, m, name });
    }
    let api_key = std::env::var("FASTGGUF_API_KEY").ok().filter(|k| !k.is_empty());

    let srv = TcpListener::bind(("0.0.0.0", port)).map_err(|e| format!("bind failed on port {port} (already in use?): {e}"))?;
    let names: Vec<String> = engs.iter().map(|e| format!("[{}]", e.name)).collect();
    println!("\n=== fastgguf-rs server (multi-model) ===");
    println!("  models : {}", names.join(" "));
    println!("  kernels: {} | {} threads", crate::simd::cpu_name(), rayon::current_num_threads());
    println!("  web UI : http://localhost:{port}/");
    println!("  API    : http://<your-ip>:{port}/v1/chat/completions (OpenAI-compatible)");
    println!("  modes  : \"model\":\"<name>\" | \"mode\":\"parallel\" | \"mode\":\"chain\" | \"mode\":\"loop\" (+\"rounds\":N)");
    println!(
        "  auth   : {}",
        if api_key.is_some() { "API key required (FASTGGUF_API_KEY)" } else { "open (set FASTGGUF_API_KEY to protect)" }
    );
    println!("  ctx    : {n_ctx} tokens/model | Ctrl+C to stop\n");

    let models_json = format!(
        "{{\"object\":\"list\",\"data\":[{}]}}",
        engs.iter().map(|e| format!("{{\"id\":{},\"object\":\"model\"}}", quote(&e.name))).collect::<Vec<_>>().join(",")
    );

    // one request at a time: generation uses every core anyway
    for conn in srv.incoming() {
        let Ok(mut c) = conn else { continue };
        let _ = c.set_nodelay(true);
        let _ = c.set_read_timeout(Some(Duration::from_secs(30)));
        let req = match read_request(&mut c) {
            Ok(r) => r,
            Err(413) => {
                json_err(&mut c, 413, "request too large");
                continue;
            }
            Err(_) => continue,
        };
        match (req.method.as_str(), req.url.as_str()) {
            ("OPTIONS", _) => http_reply(&mut c, 200, "text/plain", b""),
            ("GET", "/") | ("GET", "/index.html") => http_reply(&mut c, 200, "text/html; charset=utf-8", INDEX_HTML.as_bytes()),
            ("GET", "/v1/models") => http_reply(&mut c, 200, "application/json", models_json.as_bytes()),
            ("POST", "/v1/chat/completions") => {
                if let Some(key) = &api_key {
                    let ok = req.headers.iter().any(|(k, v)| k == "authorization" && *v == format!("Bearer {key}"));
                    if !ok {
                        json_err(&mut c, 401, "invalid api key");
                        continue;
                    }
                }
                handle_chat(&mut c, &mut engs, &String::from_utf8_lossy(&req.body));
            }
            _ => json_err(&mut c, 404, "not found"),
        }
        let _ = c.flush();
    }
    Ok(())
}

// built-in chat page: model and mode selection, streaming replies
const INDEX_HTML: &str = r##"<!DOCTYPE html><html><head><meta charset=utf-8>
<meta name=viewport content='width=device-width,initial-scale=1'>
<title>fastgguf</title><style>
*{box-sizing:border-box;margin:0}body{font-family:system-ui,'Segoe UI',sans-serif;background:#0f1117;color:#e2e4ea;height:100vh;display:flex;flex-direction:column}
header{padding:12px 20px;background:#161a24;border-bottom:1px solid #262b38;font-weight:600;display:flex;justify-content:space-between;align-items:center}
header span{color:#7a8399;font-size:13px;font-weight:400}
select{background:#0f1117;color:#e2e4ea;border:1px solid #2c3242;border-radius:8px;padding:5px 8px;font-size:13px;outline:none}select:disabled{opacity:.45}
#ctl{display:flex;gap:8px;align-items:center}
#log{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:12px}
.m{max-width:78%;padding:10px 14px;border-radius:14px;white-space:pre-wrap;word-break:break-word;line-height:1.55;unicode-bidi:plaintext}
.u{align-self:flex-end;background:#2b5cd9;border-bottom-right-radius:4px}
.a{align-self:flex-start;background:#1c212e;border-bottom-left-radius:4px}
footer{padding:14px 20px;background:#161a24;border-top:1px solid #262b38;display:flex;gap:10px}
#in{flex:1;background:#0f1117;border:1px solid #2c3242;border-radius:10px;color:#e2e4ea;padding:11px 14px;font-size:15px;outline:none}#in:focus{border-color:#2b5cd9}
button{background:#2b5cd9;border:none;border-radius:10px;color:#fff;padding:0 22px;font-size:15px;cursor:pointer}button:disabled{opacity:.45}
</style></head><body>
<header>fastgguf<div id=ctl>
<select id=md><option value=single>single</option><option value=parallel>parallel</option><option value=chain>chain</option><option value=loop>loop</option></select>
<input id=rn type=number value=2 min=1 max=8 title=rounds style='width:52px;display:none;background:#0f1117;color:#e2e4ea;border:1px solid #2c3242;border-radius:8px;padding:5px 8px;font-size:13px'>
<select id=mo></select><span id=st>ready</span></div></header>
<div id=log></div>
<footer><input id=in placeholder='اكتب رسالتك... / Type a message' autofocus>
<button id=btn>&#10148;</button></footer><script>
const log=document.getElementById('log'),inp=document.getElementById('in'),btn=document.getElementById('btn'),st=document.getElementById('st'),mo=document.getElementById('mo'),md=document.getElementById('md');let hist=[];
fetch('/v1/models').then(r=>r.json()).then(j=>{j.data.forEach(m=>{const o=document.createElement('option');o.value=m.id;o.textContent=m.id;mo.appendChild(o)});
if(j.data.length<2){mo.style.display='none';for(const o of md.options)if(o.value==='parallel'||o.value==='chain')o.disabled=true}}).catch(()=>{});
md.onchange=()=>{mo.disabled=md.value!=='single';document.getElementById('rn').style.display=md.value==='loop'?'':'none'};
function add(cls,txt){const d=document.createElement('div');d.className='m '+cls;d.textContent=txt;log.appendChild(d);log.scrollTop=log.scrollHeight;return d}
async function go(){const q=inp.value.trim();if(!q)return;inp.value='';btn.disabled=true;
add('u',q);hist.push({role:'user',content:q});const el=add('a','');st.textContent='thinking...';
try{const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({messages:hist,stream:true,max_tokens:1024,model:mo.value,mode:md.value,rounds:+document.getElementById('rn').value||2})});
const rd=r.body.getReader(),dec=new TextDecoder();let buf='',out='';
for(;;){const{done,value}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});
let i;while((i=buf.indexOf('\n\n'))>=0){const ln=buf.slice(0,i).trim();buf=buf.slice(i+2);
if(!ln.startsWith('data:'))continue;const d=ln.slice(5).trim();if(d==='[DONE]')continue;
try{const j=JSON.parse(d);const c=j.choices[0].delta.content;if(c){out+=c;el.textContent=out;log.scrollTop=log.scrollHeight}}catch(e){}}}
hist.push({role:'assistant',content:out});st.textContent='ready'}
catch(e){el.textContent='[error] '+e;st.textContent='error'}btn.disabled=false;inp.focus()}
btn.onclick=go;inp.onkeydown=e=>{if(e.key==='Enter')go()};
</script></body></html>"##;
