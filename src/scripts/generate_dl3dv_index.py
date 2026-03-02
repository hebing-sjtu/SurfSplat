import json
from pathlib import Path

import torch
from tqdm import tqdm

DATASET_PATH = Path("/media/ps/ssd6/bing/scannet_chunks")

if __name__ == "__main__":
    # "train" or "test"
    for stage in ["test"]:
        stage = DATASET_PATH / stage

        index = {}
        for chunk_path in tqdm(
            sorted(list(stage.iterdir())), desc=f"Indexing {stage.name}"
        ):
            if chunk_path.suffix == ".torch":
                chunk = torch.load(chunk_path)
                for example in chunk:
                    index[example["key"]] = str(chunk_path.relative_to(stage))
        with (stage / "index.json").open("w") as f:
            json.dump(index, f)
