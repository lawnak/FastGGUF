fastgguf - run GGUF language models locally

Windows:
  1. Download a .gguf model (for example Qwen2.5 1.5B Instruct Q4_K_M from Hugging Face).
  2. Drag the .gguf file onto chat.bat   -> chat in a terminal window
     or drag it onto server.bat          -> web chat opens at http://localhost:8080
  3. Optional check:  fastgguf.exe selftest

Commands:
  fastgguf chat    model.gguf [ctx]
  fastgguf serve   model.gguf [model2.gguf ...] [port] [ctx]
  fastgguf inspect model.gguf
  fastgguf gpu
  fastgguf selftest

The "compat" build is for old CPUs without AVX2 (before 2013); it is much slower.
Full documentation: https://github.com/REPO
