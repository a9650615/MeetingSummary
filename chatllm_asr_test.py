import os, sys, time
_C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatllm.cpp")
sys.argv[0] = os.path.join(_C, "scripts", "_x.py")  # chatllm derives PATH_APP from argv[0]
sys.path.insert(0, os.path.join(_C, "bindings"))
sys.path.insert(0, os.path.join(_C, "scripts"))
from chatllm import LibChatLLM, ChatLLM


class Cap(ChatLLM):
    acc = ""
    def callback_print(self, s):
        self.acc += s
    def callback_print_meta(self, s):
        pass


lib = LibChatLLM(os.path.join(_C, "bindings"))
t0 = time.time()
m = Cap(lib, ["-m", ":qwen3-asr", "-ngl", "999"])  # all layers on Metal
print("LOAD %.1fs" % (time.time() - t0))
for i in range(2):
    m.acc = ""
    m.restart()
    t0 = time.time()
    m.chat([{"type": "audio", "file": "/tmp/sample.wav"}])
    print("RUN%d %.1fs: %s" % (i, time.time() - t0, m.acc.strip()[:60]))
