# Lint as: python3
import json
import logging
import os
import numpy as np
from PIL import Image
import datasets
from transformers import AutoTokenizer
# import torch
# from torchvision import transforms
# from xfund.image_utils import Compose, RandomResizedCropAndInterpolationWithTwoPic


#_URL = "https://github.com/doc-analysis/XFUND/releases/download/v1.0/"
_PATH = "xfund/"
_LANG = ["zh"]
_DESCRIPTION = "XFUND_zh"
logger = logging.getLogger(__name__)

XFUND_label2ids = {
    "O":0,
    'B-HEADER':1,
    'I-HEADER':2,
    'B-QUESTION':3,
    'I-QUESTION':4,
    'B-ANSWER':5,
    'I-ANSWER':6,
}


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def simplify_bbox(bbox):
    return [
        min(bbox[0::2]),
        min(bbox[1::2]),
        max(bbox[2::2]),
        max(bbox[3::2]),
    ]


def merge_bbox(bbox_list):
    x0, y0, x1, y1 = list(zip(*bbox_list))
    return [min(x0), min(y0), max(x1), max(y1)]


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    # resize image to 224x224
    image = image.resize((224, 224))
    image = np.asarray(image)  
    image = image[:, :, ::-1] # flip color channels from RGB to BGR
    image = image.transpose(2, 0, 1) # move channels to first dimension
    return image, (w, h)

class XFUNDConfig(datasets.BuilderConfig):
    """BuilderConfig for XFUND."""

    def __init__(self, lang, additional_langs=None, **kwargs):
        """
        Args:
            lang: string, language for the input text
            **kwargs: keyword arguments forwarded to super.
        """
        super(XFUNDConfig, self).__init__(**kwargs)
        self.lang = lang
        self.additional_langs = additional_langs


class XFUND(datasets.GeneratorBasedBuilder):
    """XFUND dataset."""

    BUILDER_CONFIGS = [XFUNDConfig(name=f"xfund.{lang}", lang=lang) for lang in _LANG]

    #tokenizer = AutoTokenizer.from_pretrained("./model/")

    def box_norm(self, box, width, height):
        def clip(min_num, num, max_num):
            return min(max(num, min_num), max_num)

        x0, y0, x1, y1 = box
        x0 = clip(0, int((x0 / width) * 1000), 1000)
        y0 = clip(0, int((y0 / height) * 1000), 1000)
        x1 = clip(0, int((x1 / width) * 1000), 1000)
        y1 = clip(0, int((y1 / height) * 1000), 1000)
        assert x1 >= x0
        assert y1 >= y0
        return [x0, y0, x1, y1]

    def get_segment_ids(self, bboxs):
        segment_ids = []
        for i in range(len(bboxs)):
            if i == 0:
                segment_ids.append(0)
            else:
                if bboxs[i - 1] == bboxs[i]:
                    segment_ids.append(segment_ids[-1])
                else:
                    segment_ids.append(segment_ids[-1] + 1)
        return segment_ids

    def get_position_ids(self, segment_ids):
        position_ids = []
        for i in range(len(segment_ids)):
            if i == 0:
                position_ids.append(2)
            else:
                if segment_ids[i] == segment_ids[i - 1]:
                    position_ids.append(position_ids[-1] + 1)
                else:
                    position_ids.append(2)
        return position_ids
    
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.ClassLabel(
                            names=["O", "B-QUESTION", "B-ANSWER", "B-HEADER", "I-ANSWER", "I-QUESTION", "I-HEADER"]
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        #downloaded_files = dl_manager.download_and_extract(urls_to_download)
        train_files = [f"{_PATH}{self.config.lang}.train.json", f"{_PATH}{self.config.lang}.train"]
        val_files = [f"{_PATH}{self.config.lang}.val.json", f"{_PATH}{self.config.lang}.val"]

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": train_files}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepaths": val_files}
            ),
            # datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": test_files_for_many_langs}),
        ]

    def _generate_examples(self, filepaths):
        #self.label2ids = XFUND_label2ids
        logger.info("Generating examples from = %s", filepaths)
        # open json
        with open(filepaths[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        # re-org data format
        for doc in data["documents"]:
            total_data = {"words": [], "bboxes": [], "ner_tags": []}        
            width, height = doc['img']['width'], doc['img']['height']
            doc["img"]["fpath"] = os.path.join(filepaths[1], doc["img"]["fname"])
            image, _ = load_image(doc["img"]["fpath"])
            cur_doc_words, cur_doc_bboxes, cur_doc_ner_tags, cur_doc_image_path = [], [], [], []
            for j in range(len(doc['document'])):
                # 每一行是一个分词
                cur_item = doc['document'][j]
                cur_label = cur_item['label'].upper()
                cur_word = cur_item['text'] # Text是完整的句子，字符在words中
                for k in range(len(cur_item['words'])):
                    cur_ch = cur_item["words"][k]
                    cur_doc_words.append(cur_ch['text']) # Text是完整的句子，字符在words中
                    cur_doc_bboxes.append(self.box_norm(cur_ch['box'], width=width, height=height))

                #拆分label
                if cur_label == 'OTHER':
                    cur_labels = ["O"] * len(cur_word)
                else:
                    cur_labels = [cur_label] * len(cur_word)
                    cur_labels[0] = 'B-' + cur_labels[0] # 第一个作为begin
                    for k in range(1, len(cur_labels)):
                        cur_labels[k] = 'I-' + cur_labels[k]
                cur_doc_ner_tags.extend(cur_labels)

            total_data['words'] += cur_doc_words
            total_data['bboxes'] += cur_doc_bboxes
            total_data['ner_tags'] += cur_doc_ner_tags

            chunk_size = 512
            for chunk_id, index in enumerate(range(0, len(cur_doc_words), chunk_size)):
                item = {}
                for k in total_data:
                    item[k] = total_data[k][index:index + chunk_size]

                item.update({
                    "id": f"{doc['id']}_{chunk_id}",
                    "image": image,
                    #'image_path': doc['img']['fname'],
                })
                yield f"{doc['id']}_{chunk_id}", item

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')