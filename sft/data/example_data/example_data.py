import json
import datasets
from typing import Any, Dict, List


_DESCRIPTION = "KnowCoder SFT Dataset."
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = ""
_URL = "example_data.jsonl"

need_fields = set(['one-stage.zero-shot.prompt', 'one-stage.output'])


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class ExampleDataset(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.0")

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "one-stage.zero-shot.prompt": datasets.Value("string"),
            "one-stage.output": datasets.Value("string"),
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        file_path = dl_manager.download(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": file_path
                }
            )
        ]

    def _generate_examples(self, filepath: str) -> Dict[int, Dict[str, Any]]:
        example_dataset = []
        with open(filepath, 'r', encoding='UTF-8') as file:
            for line in file:
                example = flatten_dict(json.loads(line.strip('\n')))
                example = {k: v for k, v in example.items() if k in need_fields}
                example_dataset.append(example)
        for key, example in enumerate(example_dataset):
            yield key, example

