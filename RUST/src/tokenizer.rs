// Tokenizer — SentencePiece (SPM) or GPT-2 BPE, greedy longest-prefix match, plus the chat
// templates (role tokens are added by id, never matched from text).

use crate::gguf::{Gguf, KvVal};
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tmpl {
    Gemma,
    ChatMl,
    Mistral,
    Llama3,
    Gemma4,
}

impl Tmpl {
    pub fn name(self) -> &'static str {
        match self {
            Tmpl::Gemma => "gemma",
            Tmpl::ChatMl => "chatml",
            Tmpl::Mistral => "mistral",
            Tmpl::Llama3 => "llama3",
            Tmpl::Gemma4 => "gemma4",
        }
    }
}

const TOKEN_TYPE_CONTROL: i32 = 3;

#[derive(Clone)]
pub struct Msg {
    pub role: String,
    pub content: String,
}

pub struct Tokenizer {
    pub tokens: Vec<Vec<u8>>,
    map: HashMap<Box<[u8]>, i32>,
    pub bos_id: i32,
    pub stop_ids: Vec<i32>,
    byte_tok: [i32; 256], // SPM: <0xXX> → id (or -1)
    pub is_bpe: bool,     // GPT-2 BPE (Qwen, Llama 3) or SentencePiece (Gemma, Mistral)
    pub add_bos: bool,
    pub tmpl: Tmpl,
    ttype: Vec<i32>,
    byte_cp: [u16; 256], // BPE: byte → codepoint (GPT-2 map)
    cp_byte: [i16; 512], // BPE: codepoint → byte (or -1)
}

fn parse_byte_token(s: &[u8]) -> Option<u8> {
    // "<0xXX>"
    if s.len() == 6 && s.starts_with(b"<0x") && s[5] == b'>' {
        let h = std::str::from_utf8(&s[3..5]).ok()?;
        return u8::from_str_radix(h, 16).ok();
    }
    None
}

/// first UTF-8 code point of s and its length in bytes (a broken byte counts as one)
fn utf8_decode(s: &[u8]) -> (u32, usize) {
    let c = s[0];
    let cont = |i: usize| s.get(i).is_some_and(|b| b & 0xC0 == 0x80);
    if c < 0x80 {
        (c as u32, 1)
    } else if c >> 5 == 6 && cont(1) {
        (((c & 0x1F) as u32) << 6 | (s[1] & 0x3F) as u32, 2)
    } else if c >> 4 == 14 && cont(1) && cont(2) {
        (((c & 0x0F) as u32) << 12 | ((s[1] & 0x3F) as u32) << 6 | (s[2] & 0x3F) as u32, 3)
    } else if c >> 3 == 30 && cont(1) && cont(2) && cont(3) {
        (
            ((c & 0x07) as u32) << 18 | ((s[1] & 0x3F) as u32) << 12 | ((s[2] & 0x3F) as u32) << 6 | (s[3] & 0x3F) as u32,
            4,
        )
    } else {
        (c as u32, 1)
    }
}

impl Tokenizer {
    pub fn load(g: &Gguf) -> Result<Tokenizer, String> {
        let tokens = match g.kv("tokenizer.ggml.tokens") {
            Some(KvVal::StrArr(v)) => v.clone(),
            _ => return Err("tokenizer.ggml.tokens not found".into()),
        };
        let is_bpe = g.str("tokenizer.ggml.model") == Some("gpt2");
        let ttype = match g.kv("tokenizer.ggml.token_type") {
            Some(KvVal::IntArr(v)) => v.clone(),
            _ => Vec::new(),
        };
        let mut map = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            // with duplicates the lowest id wins, like the C hash map
            map.entry(t.clone().into_boxed_slice()).or_insert(i as i32);
        }
        let mut t = Tokenizer {
            tokens,
            map,
            bos_id: -1,
            stop_ids: Vec::new(),
            byte_tok: [-1; 256],
            is_bpe,
            add_bos: false,
            tmpl: Tmpl::Mistral,
            ttype,
            byte_cp: [0; 256],
            cp_byte: [-1; 512],
        };
        if is_bpe {
            // GPT-2 map: printable bytes stay as they are, the rest move to 256+
            let mut n = 0u16;
            for b in 0..256usize {
                let keep = (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
                let cp = if keep {
                    b as u16
                } else {
                    n += 1;
                    255 + n
                };
                t.byte_cp[b] = cp;
                t.cp_byte[cp as usize] = b as i16;
            }
        }

        t.bos_id = g.num("tokenizer.ggml.bos_token_id", -1.0) as i32;
        if t.bos_id < 0 {
            t.bos_id = t.lookup(b"<bos>");
        }
        t.add_bos = g.num("tokenizer.ggml.add_bos_token", if is_bpe { 0.0 } else { 1.0 }) != 0.0;
        if t.bos_id < 0 {
            t.add_bos = false;
        }

        let eos = g.num("tokenizer.ggml.eos_token_id", -1.0) as i32;
        if eos >= 0 {
            t.stop_ids.push(eos); // from metadata: always accepted
        }
        for name in ["<eos>", "<end_of_turn>", "<turn|>", "<|im_end|>", "<|eot_id|>", "<|endoftext|>", "</s>"] {
            let id = t.lookup(name.as_bytes());
            t.add_stop(id);
        }

        if !is_bpe {
            for (i, s) in t.tokens.iter().enumerate() {
                if let Some(b) = parse_byte_token(s) {
                    t.byte_tok[b as usize] = i as i32;
                }
            }
        }

        // the chat template: from tokenizer.chat_template first, then from the vocabulary
        let cts = g.str("tokenizer.chat_template").unwrap_or("");
        let has = |s: &str| cts.contains(s);
        t.tmpl = if has("<|turn>") {
            Tmpl::Gemma4
        } else if has("<start_of_turn>") {
            Tmpl::Gemma
        } else if has("<|im_start|>") {
            Tmpl::ChatMl
        } else if has("<|start_header_id|>") {
            Tmpl::Llama3
        } else if has("[INST]") {
            Tmpl::Mistral
        } else if t.lookup(b"<|turn>") >= 0 {
            Tmpl::Gemma4
        } else if t.lookup(b"<start_of_turn>") >= 0 {
            Tmpl::Gemma
        } else if t.lookup(b"<|im_start|>") >= 0 {
            Tmpl::ChatMl
        } else if t.lookup(b"<|start_header_id|>") >= 0 {
            Tmpl::Llama3
        } else if t.lookup(b"[INST]") >= 0 {
            Tmpl::Mistral
        } else {
            let arch = g.str("general.architecture").unwrap_or("");
            if arch.starts_with("gemma") {
                Tmpl::Gemma
            } else if arch.starts_with("qwen") {
                Tmpl::ChatMl
            } else {
                Tmpl::Mistral
            }
        };
        // the Llama 3 template always starts with <|begin_of_text|>; many files do not set add_bos_token
        if t.tmpl == Tmpl::Llama3 && t.bos_id >= 0 && g.kv("tokenizer.ggml.add_bos_token").is_none() {
            t.add_bos = true;
        }

        let stops: Vec<String> = t.stop_ids.iter().map(|i| i.to_string()).collect();
        println!(
            "[Tokenizer] {} tokens ({}), template={}, bos={} add_bos={}, stops={} ",
            t.tokens.len(),
            if is_bpe { "BPE" } else { "SPM" },
            t.tmpl.name(),
            t.bos_id,
            t.add_bos as i32,
            stops.join(" ")
        );
        Ok(t)
    }

    /// an empty tokenizer (self-test of the sampler)
    pub fn empty() -> Tokenizer {
        Tokenizer {
            tokens: Vec::new(),
            map: HashMap::new(),
            bos_id: -1,
            stop_ids: Vec::new(),
            byte_tok: [-1; 256],
            is_bpe: false,
            add_bos: false,
            tmpl: Tmpl::Mistral,
            ttype: Vec::new(),
            byte_cp: [0; 256],
            cp_byte: [-1; 512],
        }
    }

    pub fn lookup(&self, s: &[u8]) -> i32 {
        self.map.get(s).copied().unwrap_or(-1)
    }

    fn is_control(&self, id: i32) -> bool {
        id >= 0 && (id as usize) < self.ttype.len() && self.ttype[id as usize] == TOKEN_TYPE_CONTROL
    }

    fn add_stop(&mut self, id: i32) {
        if id < 0 {
            return;
        }
        // a token found by text lookup counts only if it is CONTROL — otherwise it may be ordinary text
        if (id as usize) < self.ttype.len() && self.ttype[id as usize] != TOKEN_TYPE_CONTROL {
            return;
        }
        if !self.stop_ids.contains(&id) && self.stop_ids.len() < 8 {
            self.stop_ids.push(id);
        }
    }

    pub fn is_stop(&self, id: i32) -> bool {
        self.stop_ids.contains(&id)
    }

    /// SPM: space → U+2581, BPE: the GPT-2 byte → code point map; then greedy longest match
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<i32> {
        let src = text.as_bytes();
        let mut buf = Vec::with_capacity(src.len() * 3);
        if self.is_bpe {
            for &c in src {
                let cp = self.byte_cp[c as usize] as u32;
                if cp < 0x80 {
                    buf.push(cp as u8);
                } else {
                    buf.push((0xC0 | (cp >> 6)) as u8);
                    buf.push((0x80 | (cp & 0x3F)) as u8);
                }
            }
        } else {
            for &c in src {
                if c == b' ' {
                    buf.extend_from_slice(&[0xE2, 0x96, 0x81]);
                } else {
                    buf.push(c);
                }
            }
        }
        let mut ids = Vec::with_capacity(buf.len() + 2);
        if add_bos && self.bos_id >= 0 {
            ids.push(self.bos_id);
        }
        let mut i = 0;
        while i < buf.len() {
            let max_l = (buf.len() - i).min(64);
            let mut found = None;
            for l in (1..=max_l).rev() {
                let id = self.lookup(&buf[i..i + l]);
                // control tokens never come from text: the template adds them by id, otherwise
                // the user could break the template by typing them
                if id >= 0 && !self.is_control(id) {
                    found = Some((id, l));
                    break;
                }
            }
            match found {
                Some((id, l)) => {
                    ids.push(id);
                    i += l;
                }
                None if self.is_bpe => i += if buf[i] >= 0xC0 { 2 } else { 1 },
                None => {
                    let b = self.byte_tok[buf[i] as usize];
                    if b >= 0 {
                        ids.push(b);
                    }
                    i += 1;
                }
            }
        }
        ids
    }

    fn push_text(&self, out: &mut Vec<i32>, a: &str, b: &str) {
        if a.is_empty() && b.is_empty() {
            return;
        }
        out.extend(self.encode(&format!("{a}{b}"), false));
    }

    /// special token by name: by id when the vocabulary has it, otherwise as text
    fn push_special(&self, out: &mut Vec<i32>, name: &str) {
        let id = self.lookup(name.as_bytes());
        if id >= 0 {
            out.push(id);
        } else {
            self.push_text(out, name, "");
        }
    }

    /// encodes the conversation with the model's template and opens the assistant turn.
    /// close_prev: first close the model's previous reply (console mode: the stop token that
    /// ended it was never passed through the model)
    pub fn encode_chat(&self, msgs: &[Msg], add_bos: bool, close_prev: bool) -> Vec<i32> {
        let mut b = Vec::new();
        if add_bos && self.bos_id >= 0 {
            b.push(self.bos_id);
        }
        match self.tmpl {
            Tmpl::ChatMl => {
                if close_prev {
                    self.push_special(&mut b, "<|im_end|>");
                    self.push_text(&mut b, "\n", "");
                }
                for m in msgs {
                    let role: String = m.role.chars().take(15).collect();
                    self.push_special(&mut b, "<|im_start|>");
                    self.push_text(&mut b, &format!("{role}\n"), &m.content);
                    self.push_special(&mut b, "<|im_end|>");
                    self.push_text(&mut b, "\n", "");
                }
                self.push_special(&mut b, "<|im_start|>");
                self.push_text(&mut b, "assistant\n", "");
            }
            Tmpl::Llama3 => {
                if close_prev {
                    self.push_special(&mut b, "<|eot_id|>");
                }
                for m in msgs {
                    self.push_special(&mut b, "<|start_header_id|>");
                    self.push_text(&mut b, &m.role, "");
                    self.push_special(&mut b, "<|end_header_id|>");
                    self.push_text(&mut b, "\n\n", &m.content);
                    self.push_special(&mut b, "<|eot_id|>");
                }
                self.push_special(&mut b, "<|start_header_id|>");
                self.push_text(&mut b, "assistant", "");
                self.push_special(&mut b, "<|end_header_id|>");
                self.push_text(&mut b, "\n\n", "");
            }
            Tmpl::Mistral => {
                if close_prev {
                    self.push_special(&mut b, "</s>");
                }
                // system is merged into the first user message that follows
                let mut sys: Option<&str> = None;
                for m in msgs {
                    if m.role == "system" {
                        sys = Some(&m.content);
                        continue;
                    }
                    if m.role == "assistant" {
                        self.push_text(&mut b, " ", &m.content);
                        self.push_special(&mut b, "</s>");
                    } else {
                        self.push_special(&mut b, "[INST]");
                        if let Some(s) = sys.take() {
                            self.push_text(&mut b, " ", s);
                            self.push_text(&mut b, "\n\n", &m.content);
                        } else {
                            self.push_text(&mut b, " ", &m.content);
                        }
                        self.push_text(&mut b, " ", "");
                        self.push_special(&mut b, "[/INST]");
                    }
                }
            }
            Tmpl::Gemma4 => {
                if close_prev {
                    self.push_special(&mut b, "<turn|>");
                    self.push_text(&mut b, "\n", "");
                }
                for m in msgs {
                    let role = match m.role.as_str() {
                        "assistant" => "model\n",
                        "system" => "system\n",
                        _ => "user\n",
                    };
                    self.push_special(&mut b, "<|turn>");
                    self.push_text(&mut b, role, &m.content);
                    self.push_special(&mut b, "<turn|>");
                    self.push_text(&mut b, "\n", "");
                }
                self.push_special(&mut b, "<|turn>");
                self.push_text(&mut b, "model\n", "");
            }
            Tmpl::Gemma => {
                // no separate system role: it becomes a user turn
                if close_prev {
                    self.push_special(&mut b, "<end_of_turn>");
                    self.push_text(&mut b, "\n", "");
                }
                for m in msgs {
                    let role = if m.role == "assistant" { "model\n" } else { "user\n" };
                    self.push_special(&mut b, "<start_of_turn>");
                    self.push_text(&mut b, role, &m.content);
                    self.push_special(&mut b, "<end_of_turn>");
                    self.push_text(&mut b, "\n", "");
                }
                self.push_special(&mut b, "<start_of_turn>");
                self.push_text(&mut b, "model\n", "");
            }
        }
        b
    }

    /// next user turn in console mode; the first turn (pos 0) starts with BOS, later ones close the previous reply
    pub fn encode_turn(&self, msg: &str, pos: usize) -> Vec<i32> {
        let one = [Msg { role: "user".into(), content: msg.into() }];
        self.encode_chat(&one, pos == 0 && self.add_bos, pos > 0)
    }

    /// a non-printing control token? (token_type CONTROL or the special-token shape)
    fn is_special(&self, id: i32) -> bool {
        if self.is_control(id) {
            return true;
        }
        let w = &self.tokens[id as usize];
        let n = w.len();
        if self.is_bpe {
            n >= 4 && w[0] == b'<' && w[1] == b'|' && w[n - 2] == b'|' && w[n - 1] == b'>'
        } else {
            n >= 2 && w[0] == b'<' && w[n - 1] == b'>'
        }
    }

    /// the bytes a token stands for (empty for control tokens)
    pub fn token_bytes(&self, id: i32) -> Vec<u8> {
        if id < 0 || id as usize >= self.tokens.len() {
            return Vec::new();
        }
        let w = &self.tokens[id as usize];
        if !self.is_bpe {
            if let Some(b) = parse_byte_token(w) {
                return vec![b];
            }
        }
        if self.is_special(id) {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(w.len());
        if self.is_bpe {
            // every mapped code point goes back to its original byte
            let mut i = 0;
            while i < w.len() {
                let (cp, l) = utf8_decode(&w[i..]);
                if cp < 512 && self.cp_byte[cp as usize] >= 0 {
                    out.push(self.cp_byte[cp as usize] as u8);
                } else {
                    out.extend_from_slice(&w[i..i + l]);
                }
                i += l;
            }
            return out;
        }
        let mut i = 0;
        while i < w.len() {
            // U+2581 (E2 96 81) → space
            if i + 2 < w.len() && w[i] == 0xE2 && w[i + 1] == 0x96 && w[i + 2] == 0x81 {
                out.push(b' ');
                i += 3;
            } else {
                out.push(w[i]);
                i += 1;
            }
        }
        out
    }
}

/// Streams token bytes to stdout. Byte tokens (and BPE tokens) can split one character (e.g.
/// Arabic) across several tokens, so an incomplete UTF-8 sequence is held back until it is whole.
pub struct Utf8Out {
    pend: Vec<u8>,
    need: usize,
}

impl Utf8Out {
    pub fn new() -> Self {
        Utf8Out { pend: Vec::with_capacity(4), need: 0 }
    }

    /// bytes ready to print after adding these
    pub fn push(&mut self, bytes: &[u8], out: &mut Vec<u8>) {
        for &c in bytes {
            if self.need > 0 {
                if c & 0xC0 == 0x80 {
                    self.pend.push(c);
                    if self.pend.len() == self.need {
                        out.append(&mut self.pend);
                        self.need = 0;
                    }
                    continue;
                }
                // broken sequence: print it as it is and go on
                out.append(&mut self.pend);
                self.need = 0;
            }
            let need = if c & 0xE0 == 0xC0 {
                2
            } else if c & 0xF0 == 0xE0 {
                3
            } else if c & 0xF8 == 0xF0 {
                4
            } else {
                0
            };
            if need > 0 {
                self.pend.push(c);
                self.need = need;
            } else {
                out.push(c);
            }
        }
    }

    pub fn flush(&mut self, out: &mut Vec<u8>) {
        out.append(&mut self.pend);
        self.need = 0;
    }
}

impl Default for Utf8Out {
    fn default() -> Self {
        Self::new()
    }
}
