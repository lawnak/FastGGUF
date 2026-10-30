// Next-token selection: repeat penalty → temperature → top-k → top-p → draw.
// temp <= 0 is greedy (argmax): deterministic, useful for tests.

use crate::ops::{argmax, rng_uniform};
use crate::tokenizer::Tokenizer;

#[derive(Clone, Copy)]
pub struct Cand {
    pub v: f32,
    pub id: i32,
}

pub struct Sampler {
    pub temp: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub top_k: i32, // <= 0: no limit
    pub repeat_last_n: usize,
    pub rng: u64, // the seed / RNG state
    cand: Vec<Cand>,
}

impl Default for Sampler {
    /// Google's recommendation for Gemma: temp 1.0, top_k 64, top_p 0.95
    fn default() -> Self {
        Sampler { temp: 1.0, top_p: 0.95, repeat_penalty: 1.0, top_k: 64, repeat_last_n: 64, rng: 0, cand: Vec::new() }
    }
}

fn sort_desc(c: &mut [Cand]) {
    c.sort_unstable_by(|a, b| b.v.partial_cmp(&a.v).unwrap_or(std::cmp::Ordering::Equal));
}

/// the k largest values sorted from the top (all of them when k <= 0 or k >= n)
pub fn top_k_select(logits: &[f32], k: i32, out: &mut Vec<Cand>) -> usize {
    out.clear();
    let n = logits.len();
    if k <= 0 || k as usize >= n {
        out.extend(logits.iter().enumerate().map(|(i, &v)| Cand { v, id: i as i32 }));
        sort_desc(out);
        return n;
    }
    let k = k as usize;
    // insertion into a small sorted array: most of the vocabulary is rejected with one comparison
    for (i, &v) in logits.iter().enumerate() {
        if out.len() == k && v.partial_cmp(&out[k - 1].v) != Some(std::cmp::Ordering::Greater) {
            continue;
        }
        let mut j = if out.len() < k {
            out.push(Cand { v, id: i as i32 });
            out.len() - 1
        } else {
            k - 1
        };
        while j > 0 && out[j - 1].v < v {
            out[j] = out[j - 1];
            j -= 1;
        }
        out[j] = Cand { v, id: i as i32 };
    }
    out.len()
}

impl Sampler {
    /// hist = the conversation so far (for the repeat penalty). Changes logits in place.
    pub fn sample(&mut self, tk: &Tokenizer, logits: &mut [f32], hist: &[i32]) -> i32 {
        let n_vocab = logits.len();
        if self.repeat_penalty > 0.0 && self.repeat_penalty != 1.0 && self.repeat_last_n > 0 {
            let start = hist.len().saturating_sub(self.repeat_last_n);
            for i in start..hist.len() {
                let id = hist[i];
                // stop tokens are not penalized, otherwise the model hesitates to end its reply
                if id < 0 || id as usize >= n_vocab || tk.is_stop(id) {
                    continue;
                }
                if hist[start..i].contains(&id) {
                    continue;
                }
                let l = &mut logits[id as usize];
                *l = if *l > 0.0 { *l / self.repeat_penalty } else { *l * self.repeat_penalty };
            }
        }
        if self.temp <= 0.0 {
            return argmax(logits) as i32;
        }

        let mut c = std::mem::take(&mut self.cand);
        let n = if self.top_k > 0 {
            top_k_select(logits, self.top_k, &mut c)
        } else {
            // without top-k: ignore anything below e^-20 of the top instead of sorting the whole vocabulary
            let cut = logits[argmax(logits)] - 20.0 * self.temp;
            c.clear();
            c.extend(logits.iter().enumerate().filter(|(_, &v)| v > cut).map(|(i, &v)| Cand { v, id: i as i32 }));
            sort_desc(&mut c);
            c.len()
        };
        let mx = c[0].v;
        let mut sum = 0.0f32;
        for e in c[..n].iter_mut() {
            e.v = ((e.v - mx) / self.temp).exp();
            sum += e.v;
        }
        // top-p: the smallest set of the most likely tokens whose sum reaches top_p — at least one
        let mut keep = n;
        if self.top_p > 0.0 && self.top_p < 1.0 {
            let (mut cum, lim) = (0.0f32, self.top_p * sum);
            for (i, e) in c[..n].iter().enumerate() {
                cum += e.v;
                if cum >= lim {
                    keep = i + 1;
                    break;
                }
            }
        }
        let tot: f32 = c[..keep].iter().map(|e| e.v).sum();
        let mut r = rng_uniform(&mut self.rng) * tot;
        let mut pick = c[keep - 1].id;
        for e in &c[..keep] {
            r -= e.v;
            if r < 0.0 {
                pick = e.id;
                break;
            }
        }
        self.cand = c;
        pick
    }
}
