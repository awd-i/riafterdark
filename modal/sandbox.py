import modal, sys
app = modal.App.lookup("ripple-copy", create_if_missing=True)
vol = modal.Volume.from_name("ripple-weights")
img = (modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git","build-essential","curl","pkg-config","libssl-dev","protobuf-compiler","libnuma1","cmake")
    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal")
    .env({"PATH":"/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin","CUDA_HOME":"/usr/local/cuda"})
    .pip_install("uv")
    .run_commands("git clone --depth 1 --branch v0.5.19 https://github.com/sgl-project/sglang.git /opt/sglang",
        "cd /opt/sglang && uv pip install --system --prerelease allow --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130 -e python 'torch==2.13.0+cu130' openai tokenizers")
)
if __name__ == "__main__":
    with modal.enable_output():
        sb = modal.Sandbox.create("sleep","infinity", app=app, image=img, gpu=sys.argv[1] if len(sys.argv)>1 else "B200:2",
            volumes={"/w": vol}, timeout=4*3600, cpu=16, memory=262144)
    print("SANDBOX", sb.object_id)
