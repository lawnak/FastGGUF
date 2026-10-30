// GPU (OpenCL) — the same code runs on NVIDIA, AMD and Intel. OpenCL.dll / libOpenCL.so ships
// with every GPU driver and is loaded at run time: no SDK, no build-time dependency, and without
// a GPU or driver everything keeps running on the CPU.
// Control: FASTGGUF_GPU = auto (default: discrete GPU only) | on (integrated too) | off
//          FASTGGUF_GPU_DEVICE = device index from the [gpu] list
//          FASTGGUF_GPU_MEM = weight memory budget in GB, FASTGGUF_GPU_LAYERS = layers to offload

use crate::quant::*;
use std::ffi::{c_char, c_void, CString};
use std::ptr::null_mut;
use std::sync::atomic::{AtomicI8, Ordering};
use std::sync::Mutex;

type Ptr = *mut c_void;

const CL_SUCCESS: i32 = 0;
const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
const CL_DEVICE_NAME: u32 = 0x102B;
const CL_DEVICE_GLOBAL_MEM_SIZE: u32 = 0x101F;
const CL_DEVICE_MAX_MEM_ALLOC_SIZE: u32 = 0x1010;
const CL_DEVICE_HOST_UNIFIED_MEMORY: u32 = 0x1035;
const CL_MEM_READ_ONLY: u64 = 1 << 2;
const CL_MEM_WRITE_ONLY: u64 = 1 << 1;
const CL_PROGRAM_BUILD_LOG: u32 = 0x1183;

/// the OpenCL entry points used here
#[allow(non_snake_case)]
struct Cl {
    GetPlatformIDs: unsafe extern "system" fn(u32, *mut Ptr, *mut u32) -> i32,
    GetDeviceIDs: unsafe extern "system" fn(Ptr, u64, u32, *mut Ptr, *mut u32) -> i32,
    GetDeviceInfo: unsafe extern "system" fn(Ptr, u32, usize, *mut c_void, *mut usize) -> i32,
    CreateContext: unsafe extern "system" fn(*const isize, u32, *const Ptr, Ptr, Ptr, *mut i32) -> Ptr,
    CreateCommandQueue: unsafe extern "system" fn(Ptr, Ptr, u64, *mut i32) -> Ptr,
    CreateProgramWithSource: unsafe extern "system" fn(Ptr, u32, *const *const c_char, *const usize, *mut i32) -> Ptr,
    BuildProgram: unsafe extern "system" fn(Ptr, u32, *const Ptr, *const c_char, Ptr, Ptr) -> i32,
    GetProgramBuildInfo: unsafe extern "system" fn(Ptr, Ptr, u32, usize, *mut c_void, *mut usize) -> i32,
    CreateKernel: unsafe extern "system" fn(Ptr, *const c_char, *mut i32) -> Ptr,
    SetKernelArg: unsafe extern "system" fn(Ptr, u32, usize, *const c_void) -> i32,
    CreateBuffer: unsafe extern "system" fn(Ptr, u64, usize, Ptr, *mut i32) -> Ptr,
    EnqueueWriteBuffer: unsafe extern "system" fn(Ptr, Ptr, u32, usize, usize, *const c_void, u32, *const Ptr, *mut Ptr) -> i32,
    EnqueueReadBuffer: unsafe extern "system" fn(Ptr, Ptr, u32, usize, usize, *mut c_void, u32, *const Ptr, *mut Ptr) -> i32,
    EnqueueNDRangeKernel:
        unsafe extern "system" fn(Ptr, Ptr, u32, *const usize, *const usize, *const usize, u32, *const Ptr, *mut Ptr) -> i32,
    ReleaseMemObject: unsafe extern "system" fn(Ptr) -> i32,
}

#[cfg(windows)]
mod dl {
    use std::ffi::{c_char, c_void};
    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        fn GetProcAddress(h: *mut c_void, name: *const c_char) -> *mut c_void;
    }
    pub unsafe fn open() -> *mut c_void {
        LoadLibraryA(c"OpenCL.dll".as_ptr())
    }
    pub unsafe fn sym(h: *mut c_void, name: *const c_char) -> *mut c_void {
        GetProcAddress(h, name)
    }
}

#[cfg(unix)]
mod dl {
    use std::ffi::{c_char, c_int, c_void};
    #[cfg_attr(target_os = "linux", link(name = "dl"))]
    extern "C" {
        fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(h: *mut c_void, name: *const c_char) -> *mut c_void;
    }
    const RTLD_NOW: c_int = 2;
    pub unsafe fn open() -> *mut c_void {
        for name in [c"libOpenCL.so.1", c"libOpenCL.so", c"/System/Library/Frameworks/OpenCL.framework/OpenCL"] {
            let h = dlopen(name.as_ptr(), RTLD_NOW);
            if !h.is_null() {
                return h;
            }
        }
        std::ptr::null_mut()
    }
    pub unsafe fn sym(h: *mut c_void, name: *const c_char) -> *mut c_void {
        dlsym(h, name)
    }
}

fn load_runtime() -> Option<Cl> {
    // SAFETY: the symbols are the standard OpenCL 1.2 functions with these signatures
    unsafe {
        let h = dl::open();
        if h.is_null() {
            return None;
        }
        macro_rules! sym {
            ($name:literal) => {{
                let p = dl::sym(h, concat!("cl", $name, "\0").as_ptr() as *const c_char);
                if p.is_null() {
                    return None;
                }
                std::mem::transmute::<*mut c_void, _>(p)
            }};
        }
        Some(Cl {
            GetPlatformIDs: sym!("GetPlatformIDs"),
            GetDeviceIDs: sym!("GetDeviceIDs"),
            GetDeviceInfo: sym!("GetDeviceInfo"),
            CreateContext: sym!("CreateContext"),
            CreateCommandQueue: sym!("CreateCommandQueue"),
            CreateProgramWithSource: sym!("CreateProgramWithSource"),
            BuildProgram: sym!("BuildProgram"),
            GetProgramBuildInfo: sym!("GetProgramBuildInfo"),
            CreateKernel: sym!("CreateKernel"),
            SetKernelArg: sym!("SetKernelArg"),
            CreateBuffer: sym!("CreateBuffer"),
            EnqueueWriteBuffer: sym!("EnqueueWriteBuffer"),
            EnqueueReadBuffer: sym!("EnqueueReadBuffer"),
            EnqueueNDRangeKernel: sym!("EnqueueNDRangeKernel"),
            ReleaseMemObject: sym!("ReleaseMemObject"),
        })
    }
}

/// one kernel per quant type: out[row] = dot(W[row], x), one work-group per row
const WGS: usize = 64;
const KERNELS: [(u32, &str); 12] = [
    (T_F32, "mv_f32"),
    (T_F16, "mv_f16"),
    (T_BF16, "mv_bf16"),
    (T_Q4_0, "mv_q4_0"),
    (T_Q4_1, "mv_q4_1"),
    (T_Q5_0, "mv_q5_0"),
    (T_Q5_1, "mv_q5_1"),
    (T_Q8_0, "mv_q8_0"),
    (T_Q4_K, "mv_q4_K"),
    (T_Q5_K, "mv_q5_K"),
    (T_Q6_K, "mv_q6_K"),
    (T_IQ4_NL, "mv_iq4_nl"),
]; // Q2_K / Q3_K / IQ4_XS stay on the CPU

fn kernel_index(typ: u32) -> Option<usize> {
    KERNELS.iter().position(|(t, _)| *t == typ)
}

/// the kernel argument n: blocks per row (elements for float types)
fn n_blocks(typ: u32, cols: usize) -> Option<usize> {
    match typ {
        T_F32 | T_F16 | T_BF16 => Some(cols),
        T_Q4_0 | T_Q4_1 | T_Q5_0 | T_Q5_1 | T_Q8_0 | T_IQ4_NL if cols % 32 == 0 => Some(cols / 32),
        T_Q4_K | T_Q5_K | T_Q6_K if cols % 256 == 0 => Some(cols / 256),
        _ => None,
    }
}

/// weights uploaded to the GPU (a cl_mem handle)
pub struct GpuMem(usize);

struct Gpu {
    cl: Cl,
    ctx: Ptr,
    q: Ptr,
    k: Vec<Ptr>,
    xbuf: Ptr,
    ybuf: Ptr,
    xcap: usize,
    ycap: usize,
    vram: u64,
    max_alloc: u64,
    budget: u64,
    used: u64,
    n_offloaded: usize,
    n_skip_type: usize,
    n_skip_mem: usize,
    name: String,
}
// SAFETY: the handles are only used while holding the mutex
unsafe impl Send for Gpu {}

static STATE: AtomicI8 = AtomicI8::new(0); // 0 not initialized, 1 ready, -1 off / unavailable
static GPU: Mutex<Option<Gpu>> = Mutex::new(None);

pub fn ready() -> bool {
    STATE.load(Ordering::Relaxed) == 1
}

/// stop using the GPU (after an error): everything continues on the CPU
pub fn disable() {
    STATE.store(-1, Ordering::Relaxed);
}

struct Dev {
    id: Ptr,
    name: String,
    mem: u64,
    unified: bool,
}

unsafe fn list_devices(cl: &Cl) -> Vec<Dev> {
    let mut plats = [null_mut(); 8];
    let mut np = 0u32;
    if (cl.GetPlatformIDs)(8, plats.as_mut_ptr(), &mut np) != CL_SUCCESS {
        return Vec::new();
    }
    let mut out = Vec::new();
    for &p in &plats[..np.min(8) as usize] {
        let mut devs = [null_mut(); 8];
        let mut nd = 0u32;
        if (cl.GetDeviceIDs)(p, CL_DEVICE_TYPE_GPU, 8, devs.as_mut_ptr(), &mut nd) != CL_SUCCESS {
            continue;
        }
        for &d in &devs[..nd.min(8) as usize] {
            let mut name = [0u8; 100];
            let mut mem = 0u64;
            let mut uni = 0u32;
            (cl.GetDeviceInfo)(d, CL_DEVICE_NAME, 99, name.as_mut_ptr() as *mut c_void, null_mut());
            (cl.GetDeviceInfo)(d, CL_DEVICE_GLOBAL_MEM_SIZE, 8, &mut mem as *mut u64 as *mut c_void, null_mut());
            (cl.GetDeviceInfo)(d, CL_DEVICE_HOST_UNIFIED_MEMORY, 4, &mut uni as *mut u32 as *mut c_void, null_mut());
            let end = name.iter().position(|&c| c == 0).unwrap_or(name.len());
            out.push(Dev {
                id: d,
                name: String::from_utf8_lossy(&name[..end]).trim().to_string(),
                mem,
                unified: uni != 0,
            });
        }
    }
    out
}

/// probe = true: ignore FASTGGUF_GPU=off and accept an integrated GPU (the `gpu` command)
pub fn init(probe: bool) {
    if STATE.load(Ordering::Relaxed) != 0 {
        return;
    }
    STATE.store(-1, Ordering::Relaxed);
    let env = std::env::var("FASTGGUF_GPU").unwrap_or_default();
    let mut force_on = probe;
    if !probe {
        if matches!(env.as_str(), "off" | "0" | "cpu" | "none") {
            return;
        }
        if matches!(env.as_str(), "on" | "1" | "force" | "gpu") {
            force_on = true;
        }
    }
    let want_idx: Option<usize> = std::env::var("FASTGGUF_GPU_DEVICE").ok().and_then(|v| v.trim().parse().ok());
    if want_idx.is_some() {
        force_on = true;
    }
    let Some(cl) = load_runtime() else {
        if force_on {
            eprintln!("[gpu] OpenCL runtime not found - staying on CPU");
        }
        return;
    };
    // SAFETY: OpenCL calls with valid handles and correctly sized out-parameters
    unsafe {
        let devs = list_devices(&cl);
        if devs.is_empty() {
            if force_on {
                eprintln!("[gpu] no OpenCL GPU found - staying on CPU");
            }
            return;
        }
        for (i, d) in devs.iter().enumerate() {
            println!(
                "[gpu] device {i}: {} ({}, {} MB)",
                d.name,
                if d.unified { "integrated" } else { "discrete" },
                d.mem >> 20
            );
        }
        let biggest = |filter: &dyn Fn(&Dev) -> bool| {
            devs.iter().enumerate().filter(|(_, d)| filter(d)).max_by_key(|(_, d)| d.mem).map(|(i, _)| i)
        };
        let pick = match want_idx {
            Some(i) if i < devs.len() => i,
            _ => {
                // preference: the discrete GPU with the most memory; integrated only when asked for
                match biggest(&|d| !d.unified).or_else(|| if force_on { biggest(&|_| true) } else { None }) {
                    Some(i) => i,
                    None => {
                        println!("[gpu] only integrated GPU found - staying on CPU (set FASTGGUF_GPU=on to use it)");
                        return;
                    }
                }
            }
        };
        let dev = &devs[pick];
        let mut err = 0i32;
        let ctx = (cl.CreateContext)(std::ptr::null(), 1, &dev.id, null_mut(), null_mut(), &mut err);
        if ctx.is_null() || err != CL_SUCCESS {
            eprintln!("[gpu] context failed ({err})");
            return;
        }
        let q = (cl.CreateCommandQueue)(ctx, dev.id, 0, &mut err);
        if q.is_null() || err != CL_SUCCESS {
            eprintln!("[gpu] queue failed ({err})");
            return;
        }
        let src = CString::new(GPU_SRC).unwrap();
        let sp = src.as_ptr();
        let prog = (cl.CreateProgramWithSource)(ctx, 1, &sp, std::ptr::null(), &mut err);
        if prog.is_null() || err != CL_SUCCESS {
            eprintln!("[gpu] program failed ({err})");
            return;
        }
        let opts = CString::new(format!("-cl-mad-enable -DWGS={WGS}")).unwrap();
        if (cl.BuildProgram)(prog, 1, &dev.id, opts.as_ptr(), null_mut(), null_mut()) != CL_SUCCESS {
            let mut log = vec![0u8; 4096];
            (cl.GetProgramBuildInfo)(prog, dev.id, CL_PROGRAM_BUILD_LOG, 4095, log.as_mut_ptr() as *mut c_void, null_mut());
            let end = log.iter().position(|&c| c == 0).unwrap_or(log.len());
            eprintln!("[gpu] kernel build failed:\n{}", String::from_utf8_lossy(&log[..end]));
            return;
        }
        let mut k = Vec::new();
        for (_, name) in KERNELS {
            let n = CString::new(name).unwrap();
            let kern = (cl.CreateKernel)(prog, n.as_ptr(), &mut err);
            if kern.is_null() || err != CL_SUCCESS {
                eprintln!("[gpu] kernel {name} failed ({err})");
                return;
            }
            k.push(kern);
        }
        let mut max_alloc = 0u64;
        (cl.GetDeviceInfo)(dev.id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, 8, &mut max_alloc as *mut u64 as *mut c_void, null_mut());
        // weight memory budget: ~87% of VRAM, or FASTGGUF_GPU_MEM (GB). What does not fit stays on the CPU
        let mem_gb: f64 = std::env::var("FASTGGUF_GPU_MEM").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0.0);
        let budget = if mem_gb > 0.0 { (mem_gb * 1073741824.0) as u64 } else { dev.mem - dev.mem / 8 };
        *GPU.lock().unwrap() = Some(Gpu {
            cl,
            ctx,
            q,
            k,
            xbuf: null_mut(),
            ybuf: null_mut(),
            xcap: 0,
            ycap: 0,
            vram: dev.mem,
            max_alloc,
            budget,
            used: 0,
            n_offloaded: 0,
            n_skip_type: 0,
            n_skip_mem: 0,
            name: dev.name.clone(),
        });
        STATE.store(1, Ordering::Relaxed); // the self-check needs matvec_raw
        if let Err(e) = selfcheck() {
            eprintln!("[gpu] selfcheck FAILED on {}: {e} - staying on CPU", dev.name);
            STATE.store(-1, Ordering::Relaxed);
            return;
        }
        println!(
            "[gpu] using {} (OpenCL, {} MB vram, {} memory)",
            dev.name,
            dev.mem >> 20,
            if dev.unified { "shared" } else { "dedicated" }
        );
    }
}

/// the `gpu` command: 0 when a GPU is ready
pub fn cmd_info() -> i32 {
    init(true);
    match GPU.lock().unwrap().as_ref() {
        Some(g) if ready() => {
            println!("[gpu] READY: {}", g.name);
            0
        }
        _ => {
            println!("[gpu] no usable GPU - CPU only");
            1
        }
    }
}

/// copy weights to VRAM; None when the type has no kernel or the memory is full
pub fn upload(typ: u32, cols: usize, data: &[u8]) -> Option<GpuMem> {
    if !ready() {
        return None;
    }
    let mut lock = GPU.lock().unwrap();
    let g = lock.as_mut()?;
    if kernel_index(typ).is_none() || n_blocks(typ, cols).is_none() {
        g.n_skip_type += 1;
        return None;
    }
    let n = data.len() as u64;
    if n > g.max_alloc || g.used + n > g.budget {
        g.n_skip_mem += 1;
        return None;
    }
    // SAFETY: a fresh buffer of data.len() bytes, filled with a blocking write
    unsafe {
        let mut err = 0;
        let m = (g.cl.CreateBuffer)(g.ctx, CL_MEM_READ_ONLY, data.len(), null_mut(), &mut err);
        if m.is_null() || err != CL_SUCCESS {
            g.n_skip_mem += 1;
            return None;
        }
        if (g.cl.EnqueueWriteBuffer)(g.q, m, 1, 0, data.len(), data.as_ptr() as *const c_void, 0, std::ptr::null(), null_mut())
            != CL_SUCCESS
        {
            (g.cl.ReleaseMemObject)(m);
            g.n_skip_mem += 1;
            return None;
        }
        g.used += n;
        g.n_offloaded += 1;
        Some(GpuMem(m as usize))
    }
}

/// (tensors offloaded, tensors skipped for their type, skipped for memory, bytes used)
pub fn counters() -> (usize, usize, usize, u64) {
    GPU.lock().unwrap().as_ref().map(|g| (g.n_offloaded, g.n_skip_type, g.n_skip_mem, g.used)).unwrap_or((0, 0, 0, 0))
}

/// (device name, weight budget in bytes)
pub fn device() -> (String, u64) {
    GPU.lock().unwrap().as_ref().map(|g| (g.name.clone(), g.budget)).unwrap_or_default()
}

#[allow(dead_code)]
pub fn vram() -> u64 {
    GPU.lock().unwrap().as_ref().map(|g| g.vram).unwrap_or(0)
}

unsafe fn grow(g: &Gpu, buf: &mut Ptr, cap: &mut usize, need: usize, flags: u64) -> bool {
    if need <= *cap {
        return true;
    }
    if !buf.is_null() {
        (g.cl.ReleaseMemObject)(*buf);
    }
    let mut err = 0;
    *buf = (g.cl.CreateBuffer)(g.ctx, flags, need, null_mut(), &mut err);
    if buf.is_null() || err != CL_SUCCESS {
        *buf = null_mut();
        *cap = 0;
        return false;
    }
    *cap = need;
    true
}

fn matvec_raw(w: usize, typ: u32, stride: usize, x: &[f32], out: &mut [f32]) -> Result<(), String> {
    let (cols, rows) = (x.len(), out.len());
    let ki = kernel_index(typ).ok_or("no GPU kernel for this type")?;
    let nb = n_blocks(typ, cols).ok_or("row length not supported")? as i32;
    let mut lock = GPU.lock().unwrap();
    let g = lock.as_mut().ok_or("GPU not initialized")?;
    // SAFETY: buffers are large enough for cols / rows floats; the read is blocking
    unsafe {
        let (mut xb, mut xc, mut yb, mut yc) = (g.xbuf, g.xcap, g.ybuf, g.ycap);
        let ok = grow(g, &mut xb, &mut xc, cols * 4, CL_MEM_READ_ONLY) && grow(g, &mut yb, &mut yc, rows * 4, CL_MEM_WRITE_ONLY);
        (g.xbuf, g.xcap, g.ybuf, g.ycap) = (xb, xc, yb, yc);
        if !ok {
            return Err("buffer allocation failed".into());
        }
        let cl = &g.cl;
        let k = g.k[ki];
        let wm = w as Ptr;
        let st = stride as u64;
        let e = |c: i32, what: &str| if c == CL_SUCCESS { Ok(()) } else { Err(format!("{what} failed ({c})")) };
        e((cl.EnqueueWriteBuffer)(g.q, xb, 0, 0, cols * 4, x.as_ptr() as *const c_void, 0, std::ptr::null(), null_mut()), "write x")?;
        let p = |v: &Ptr| v as *const Ptr as *const c_void;
        e((cl.SetKernelArg)(k, 0, std::mem::size_of::<Ptr>(), p(&wm)), "arg 0")?;
        e((cl.SetKernelArg)(k, 1, 8, &st as *const u64 as *const c_void), "arg 1")?;
        e((cl.SetKernelArg)(k, 2, std::mem::size_of::<Ptr>(), p(&xb)), "arg 2")?;
        e((cl.SetKernelArg)(k, 3, std::mem::size_of::<Ptr>(), p(&yb)), "arg 3")?;
        e((cl.SetKernelArg)(k, 4, 4, &nb as *const i32 as *const c_void), "arg 4")?;
        let (gws, lws) = (rows * WGS, WGS);
        e((cl.EnqueueNDRangeKernel)(g.q, k, 1, std::ptr::null(), &gws, &lws, 0, std::ptr::null(), null_mut()), "kernel")?;
        e(
            (cl.EnqueueReadBuffer)(g.q, yb, 1, 0, rows * 4, out.as_mut_ptr() as *mut c_void, 0, std::ptr::null(), null_mut()),
            "read y",
        )?;
    }
    Ok(())
}

/// out = W·x for weights uploaded with `upload` (x.len() = cols, out.len() = rows)
pub fn matvec(m: &GpuMem, typ: u32, x: &[f32], out: &mut [f32]) -> Result<(), String> {
    matvec_raw(m.0, typ, row_stride(typ, x.len() as u64) as usize, x, out)
}

/// random rows of every type: the GPU result against the CPU
fn selfcheck() -> Result<(), String> {
    const COLS: usize = 512;
    const ROWS: usize = 3;
    let mut seed = 0x12345678ABCDEFu64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        seed >> 33
    };
    let x: Vec<f32> = (0..COLS).map(|_| (next() & 0xFFFF) as f32 / 32768.0 - 1.0).collect();
    for (typ, _) in KERNELS {
        let stride = row_stride(typ, COLS as u64) as usize;
        let mut w: Vec<u8> = (0..stride * ROWS).map(|_| next() as u8).collect();
        // random fp16 scales may be inf/NaN — write small sane values
        let (be, bb) = block_size(typ).unwrap();
        for r in 0..ROWS {
            for b in 0..COLS / be as usize {
                let p = r * stride + b * bb as usize;
                let d = 0.01 + (next() & 0xFF) as f32 / 2560.0;
                let h = f32_to_fp16(d).to_le_bytes();
                match typ {
                    T_F32 => w[p..p + 4].copy_from_slice(&d.to_le_bytes()),
                    T_BF16 => w[p..p + 2].copy_from_slice(&((d.to_bits() >> 16) as u16).to_le_bytes()),
                    T_Q4_1 | T_Q5_1 | T_Q4_K | T_Q5_K => {
                        w[p..p + 2].copy_from_slice(&h);
                        w[p + 2..p + 4].copy_from_slice(&h);
                    }
                    T_Q6_K => w[p + 208..p + 210].copy_from_slice(&h),
                    _ => w[p..p + 2].copy_from_slice(&h),
                }
            }
        }
        let mem = upload_scratch(&w)?;
        let mut got = [0.0f32; ROWS];
        let res = matvec_raw(mem, typ, stride, &x, &mut got);
        release(mem);
        res?;
        let mut deq = vec![0.0f32; COLS];
        for (r, g) in got.iter().enumerate() {
            dequantize_row(typ, &w[r * stride..(r + 1) * stride], &mut deq);
            let want = crate::matvec::dot_f32(&deq, &x);
            if (want - g).abs() > 1e-2 + 1e-3 * want.abs() {
                return Err(format!("{} row {r}: cpu={want} gpu={g}", type_name(typ)));
            }
        }
    }
    Ok(())
}

/// a temporary buffer that does not count against the weight budget (self-check)
fn upload_scratch(data: &[u8]) -> Result<usize, String> {
    let lock = GPU.lock().unwrap();
    let g = lock.as_ref().ok_or("GPU not initialized")?;
    // SAFETY: as in `upload`
    unsafe {
        let mut err = 0;
        let m = (g.cl.CreateBuffer)(g.ctx, CL_MEM_READ_ONLY, data.len(), null_mut(), &mut err);
        if m.is_null() || err != CL_SUCCESS {
            return Err(format!("buffer failed ({err})"));
        }
        if (g.cl.EnqueueWriteBuffer)(g.q, m, 1, 0, data.len(), data.as_ptr() as *const c_void, 0, std::ptr::null(), null_mut())
            != CL_SUCCESS
        {
            (g.cl.ReleaseMemObject)(m);
            return Err("upload failed".into());
        }
        Ok(m as usize)
    }
}

fn release(m: usize) {
    if let Some(g) = GPU.lock().unwrap().as_ref() {
        // SAFETY: m is a buffer created by upload_scratch
        unsafe { (g.cl.ReleaseMemObject)(m as Ptr) };
    }
}

// kernel source — the dequantization is the same as dequantize_row
const GPU_SRC: &str = r#"
inline float h2f(__global const uchar *p){ return vload_half(0,(__global const half*)p); }
void sm4(int j, __global const uchar *q, uchar *sc, uchar *m){
  if(j<4){ *sc=q[j]&63; *m=q[j+4]&63; }
  else { *sc=(uchar)((q[j+4]&0xF)|((q[j-4]>>6)<<4)); *m=(uchar)((q[j+4]>>4)|((q[j]>>6)<<4)); }
}
__constant float IQ4NL[16]={-127.f,-104.f,-83.f,-65.f,-49.f,-35.f,-22.f,-10.f,1.f,13.f,25.f,38.f,53.f,69.f,89.f,113.f};
#define ARGS __global const uchar *wv, ulong st, __global const float *x, __global float *y, int n
#define HEAD int row=get_group_id(0); int lid=get_local_id(0); __global const uchar *w=wv+(ulong)row*st; float acc=0.f;
#define TAIL __local float red[WGS]; red[lid]=acc; barrier(CLK_LOCAL_MEM_FENCE); for(int s=WGS/2;s>0;s>>=1){ if(lid<s) red[lid]+=red[lid+s]; barrier(CLK_LOCAL_MEM_FENCE); } if(lid==0) y[row]=red[0];
__kernel void mv_f32(ARGS){ HEAD __global const float *f=(__global const float*)w;
  for(int i=lid;i<n;i+=WGS) acc+=f[i]*x[i]; TAIL }
__kernel void mv_f16(ARGS){ HEAD __global const half *hh=(__global const half*)w;
  for(int i=lid;i<n;i+=WGS) acc+=vload_half(i,hh)*x[i]; TAIL }
__kernel void mv_bf16(ARGS){ HEAD __global const ushort *u=(__global const ushort*)w;
  for(int i=lid;i<n;i+=WGS) acc+=as_float((uint)u[i]<<16)*x[i]; TAIL }
__kernel void mv_q4_0(ARGS){ HEAD
  for(int b=lid;b<n;b+=WGS){ __global const uchar *p=w+b*18; float d=h2f(p); float s=0.f;
    for(int j=0;j<16;j++){ int q=p[2+j];
      s+=(float)((q&15)-8)*x[b*32+j]+(float)((q>>4)-8)*x[b*32+j+16]; }
    acc+=d*s; } TAIL }
__kernel void mv_q4_1(ARGS){ HEAD
  for(int b=lid;b<n;b+=WGS){ __global const uchar *p=w+b*20; float d=h2f(p), m=h2f(p+2); float s=0.f, sx=0.f;
    for(int j=0;j<16;j++){ int q=p[4+j]; float x0=x[b*32+j], x1=x[b*32+j+16];
      s+=(float)(q&15)*x0+(float)(q>>4)*x1; sx+=x0+x1; }
    acc+=d*s+m*sx; } TAIL }
__kernel void mv_q5_0(ARGS){ HEAD
  for(int b=lid;b<n;b+=WGS){ __global const uchar *p=w+b*22; float d=h2f(p);
    uint qh=(uint)p[2]|((uint)p[3]<<8)|((uint)p[4]<<16)|((uint)p[5]<<24); float s=0.f;
    for(int j=0;j<16;j++){ int q=p[6+j];
      int v0=(int)((q&15)|(((qh>>j)<<4)&16))-16; int v1=(int)((q>>4)|((qh>>(j+12))&16))-16;
      s+=(float)v0*x[b*32+j]+(float)v1*x[b*32+j+16]; }
    acc+=d*s; } TAIL }
__kernel void mv_q5_1(ARGS){ HEAD
  for(int b=lid;b<n;b+=WGS){ __global const uchar *p=w+b*24; float d=h2f(p), m=h2f(p+2);
    uint qh=(uint)p[4]|((uint)p[5]<<8)|((uint)p[6]<<16)|((uint)p[7]<<24); float s=0.f, sx=0.f;
    for(int j=0;j<16;j++){ int q=p[8+j]; float x0=x[b*32+j], x1=x[b*32+j+16];
      s+=(float)((q&15)|(((qh>>j)<<4)&16))*x0+(float)((q>>4)|((qh>>(j+12))&16))*x1; sx+=x0+x1; }
    acc+=d*s+m*sx; } TAIL }
__kernel void mv_q8_0(ARGS){ HEAD
  for(int b=lid;b<n;b+=WGS){ __global const uchar *p=w+b*34; float d=h2f(p);
    __global const char *q=(__global const char*)(p+2); float s=0.f;
    for(int j=0;j<32;j++) s+=(float)q[j]*x[b*32+j];
    acc+=d*s; } TAIL }
__kernel void mv_q4_K(ARGS){ HEAD
  for(int b=lid;b<n;b+=WGS){ __global const uchar *p=w+b*144; float d=h2f(p), dmin=h2f(p+2);
    __global const uchar *sc=p+4, *qs=p+16; int yb=b*256, is=0, qi=0;
    for(int j=0;j<256;j+=64){ uchar s1,m1,s2,m2; sm4(is,sc,&s1,&m1); sm4(is+1,sc,&s2,&m2);
      float t1=0.f,u1=0.f,t2=0.f,u2=0.f;
      for(int l=0;l<32;l++){ int q=qs[qi+l]; float x0=x[yb+j+l], x1=x[yb+j+l+32];
        t1+=(float)(q&15)*x0; u1+=x0; t2+=(float)(q>>4)*x1; u2+=x1; }
      acc+=d*(float)s1*t1-dmin*(float)m1*u1+d*(float)s2*t2-dmin*(float)m2*u2;
      qi+=32; is+=2; } } TAIL }
__kernel void mv_q5_K(ARGS){ HEAD
  for(int b=lid;b<n;b+=WGS){ __global const uchar *p=w+b*176; float d=h2f(p), dmin=h2f(p+2);
    __global const uchar *sc=p+4, *qh=p+16, *ql=p+48; int yb=b*256, is=0, qi=0; int hb1=1, hb2=2;
    for(int j=0;j<256;j+=64){ uchar s1,m1,s2,m2; sm4(is,sc,&s1,&m1); sm4(is+1,sc,&s2,&m2);
      float t1=0.f,u1=0.f,t2=0.f,u2=0.f;
      for(int l=0;l<32;l++){ int q=ql[qi+l]; float x0=x[yb+j+l], x1=x[yb+j+l+32];
        float h1=(qh[l]&hb1)?16.f:0.f, hx=(qh[l]&hb2)?16.f:0.f;
        t1+=((float)(q&15)+h1)*x0; u1+=x0; t2+=((float)(q>>4)+hx)*x1; u2+=x1; }
      acc+=d*(float)s1*t1-dmin*(float)m1*u1+d*(float)s2*t2-dmin*(float)m2*u2;
      qi+=32; is+=2; hb1<<=2; hb2<<=2; } } TAIL }
__kernel void mv_q6_K(ARGS){ HEAD
  for(int b=lid;b<n;b+=WGS){ __global const uchar *p=w+b*210;
    __global const uchar *ql=p, *qh=p+128; __global const char *sc=(__global const char*)(p+192);
    float d=h2f(p+208); int yb=b*256;
    for(int hf=0;hf<2;hf++){ int yo=yb+hf*128, qlo=hf*64, qho=hf*32, so=hf*8;
      for(int l=0;l<32;l++){ int is=l/16; int hv=qh[qho+l];
        int q1=(ql[qlo+l]&15)|(((hv>>0)&3)<<4); int q2=(ql[qlo+l+32]&15)|(((hv>>2)&3)<<4);
        int q3=(ql[qlo+l]>>4)|(((hv>>4)&3)<<4);  int q4=(ql[qlo+l+32]>>4)|(((hv>>6)&3)<<4);
        acc+=d*((float)sc[so+is]*(float)(q1-32)*x[yo+l]+(float)sc[so+is+2]*(float)(q2-32)*x[yo+l+32]
               +(float)sc[so+is+4]*(float)(q3-32)*x[yo+l+64]+(float)sc[so+is+6]*(float)(q4-32)*x[yo+l+96]); } } } TAIL }
__kernel void mv_iq4_nl(ARGS){ HEAD
  for(int b=lid;b<n;b+=WGS){ __global const uchar *p=w+b*18; float d=h2f(p); float s=0.f;
    for(int j=0;j<16;j++){ int q=p[2+j]; s+=IQ4NL[q&15]*x[b*32+j]+IQ4NL[q>>4]*x[b*32+j+16]; }
    acc+=d*s; } TAIL }
"#;
