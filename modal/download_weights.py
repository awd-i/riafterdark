import modal, os
app = modal.App("ripple-dl")
vol = modal.Volume.from_name("ripple-weights", create_if_missing=True)
img = modal.Image.debian_slim().pip_install("huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER":"1"})
@app.function(image=img, volumes={"/w": vol}, timeout=7200, cpu=8, memory=16384,
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN","")})])
def dl(repo: str):
    from huggingface_hub import snapshot_download
    p = snapshot_download(repo, local_dir="/w/"+repo.split("/")[1], max_workers=16)
    vol.commit(); return p
@app.local_entrypoint()
def main():
    for r in dl.map(["ricursiveAI/Ripple-65B-Preview","ricursiveAI/Ripple-65B-Preview-DFlash2"]): print(r)
