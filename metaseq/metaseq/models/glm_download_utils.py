import os
import requests

from filelock import FileLock
from tqdm import tqdm


MODEL_ULRS = {
    "bert-base-uncased": "https://cloud.tsinghua.edu.cn/f/f45fff1a308846cfa63a/?dl=1",
    "bert-large-uncased": "https://cloud.tsinghua.edu.cn/f/6d4f38c96e8c4c16917e/?dl=1",
    "roberta-base": "https://cloud.tsinghua.edu.cn/f/307fd141932440bc92da/?dl=1",
    "roberta-large": "https://cloud.tsinghua.edu.cn/f/66c42c24ca304cecaf7e/?dl=1",
    "vit-base-patch16-224-in21k": "https://cloud.tsinghua.edu.cn/f/fdf40233d9034b6a8bdc/?dl=1",
    "deit-tiny": "https://cloud.tsinghua.edu.cn/f/b759657cb80e4bc69303/?dl=1",
    "deit-small": "https://cloud.tsinghua.edu.cn/f/51498210e2c943dbbef1/?dl=1",
    "deit-base": "https://cloud.tsinghua.edu.cn/f/9a26fd1aee7146e1a848/?dl=1",
    "cait-s24-224": "https://cloud.tsinghua.edu.cn/f/bdfb12396000468b8bb9/?dl=1",
    # CLIP
    "clip": "https://cloud.tsinghua.edu.cn/f/bd29f0537f9949e6a4fb/?dl=1",  # vit-base-patch32
    "clip-vit-base-patch16": "https://lfs.aminer.cn/misc/clip/clip-vit-base-patch16.zip",
    "clip-vit-large-patch14": "https://lfs.aminer.cn/misc/clip/clip-vit-large-patch14.zip",
    "yolos-tiny": "https://cloud.tsinghua.edu.cn/f/8ee048b6a1f1403d9253/?dl=1",
    "mae-vit-base": "https://cloud.tsinghua.edu.cn/f/5ab3543f0e1d4507ad8c/?dl=1",
    "cogview-base": "https://cloud.tsinghua.edu.cn/f/df21f6d4109b4285bfd9/?dl=1",
    "glm-large-zh": "https://lfs.aminer.cn/misc/cogview/glm/glm-large-zh.zip",
    "glm-large-en-blank": "https://lfs.aminer.cn/misc/cogview/glm/glm-large-en-blank.zip",
    "glm-10b-en": "https://lfs.aminer.cn/misc/cogview/glm/glm-10b-en.zip",
    "glm-10b-zh": "https://lfs.aminer.cn/misc/cogview/glm/glm-10b-zh.zip",
    # 'glm-large-zh': 'https://cloud.tsinghua.edu.cn/f/df21f6d4109b4285bfd9/?dl=1',
    # 'glm-large-en-blank': 'https://cloud.tsinghua.edu.cn/f/df21f6d4109b4285bfd9/?dl=1',
    "gpt-neo-1.3b": "https://cloud.tsinghua.edu.cn/f/22e87976b5b745ad90af/?dl=1",
    # CogView2
    "coglm": "https://lfs.aminer.cn/misc/cogview/cogview2/coglm.zip",
    "cogview2-dsr": "https://lfs.aminer.cn/misc/cogview/cogview2/cogview2-dsr.zip",
    "cogview2-itersr": "https://lfs.aminer.cn/misc/cogview/cogview2/cogview2-itersr.zip",
    # CogVideo
    "cogvideo-stage1": "https://lfs.aminer.cn/misc/cogvideo/cogvideo-stage1.zip",
    "cogvideo-stage2": "https://lfs.aminer.cn/misc/cogvideo/cogvideo-stage2.zip",
    # DPR
    "dpr-ctx_encoder-single-nq-base": "https://cloud.tsinghua.edu.cn/f/e5475f1211a948708baa/?dl=1",
    "dpr-question_encoder-single-nq-base": "https://cloud.tsinghua.edu.cn/f/5c4aae7d11fc4c45a5bd/?dl=1",
    "dpr-reader-single-nq-base": "https://cloud.tsinghua.edu.cn/f/e169889ab40d4615a34d/?dl=1",
}


def download_with_progress_bar(save_path, url):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pbar = tqdm(total=int(r.headers["Content-Length"]), unit_scale=True)
            for chunk in r.iter_content(chunk_size=32 * 1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    pbar.update(len(chunk))


def auto_create(name, *, path=None, url=None):
    """
    Fetches the pre-trained model given by name, and downloads it to path.
    """
    if path is None:
        path = os.getenv("SAT_HOME", "~/.sat_models")  # TODO (mchoi): Rename
    path = os.path.expanduser(path)
    file_path = os.path.join(path, name + ".zip")
    model_path = os.path.join(path, name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    lock = FileLock(file_path + ".lock")
    with lock:
        if os.path.exists(file_path) or os.path.isdir(model_path):
            pass
        else:
            if url is None:
                url = MODEL_ULRS[name]
            print(f"Downloading models {url} into {file_path} ...")
            download_with_progress_bar(file_path, url)
        # unzip
        if not os.path.isdir(model_path):
            import zipfile

            print(f"Unzipping {file_path}...")
            f = zipfile.ZipFile(file_path, "r")
            f.extractall(
                path=path
            )  # TODO check hierarcy of folders and name consistency
            assert os.path.isdir(
                model_path
            ), f"Unzip failed, or the first-level folder in zip is not {name}."
    return model_path  # must return outside the `with lock` block
