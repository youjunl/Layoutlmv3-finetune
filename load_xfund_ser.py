# Lint as: python3
import json
import logging
import os
import numpy as np
from PIL import Image
import datasets
from transformers import AutoTokenizer
import torch
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

XFUND_ids2label = {
    0:"O",
    1:'B-HEADER',
    2:'I-HEADER',
    3:'B-QUESTION',
    4:'I-QUESTION',
    5:'B-ANSWER',
    6:'I-ANSWER',
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
    #image = image.resize((224, 224))
    #image = np.asarray(image)
    #image = image[:, :, ::-1] # flip color channels from RGB to BGR
    #image = image.transpose(2, 0, 1) # move channels to first dimension
    #image = torch.as_tensor(image, dtype=torch.uint8) # Convert to tensor
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

    tokenizer = AutoTokenizer.from_pretrained("./model/")

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
    
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "input_ids": datasets.Sequence(datasets.Value("int64")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "labels": datasets.Sequence(
                        datasets.ClassLabel(
                            names=["O", "B-QUESTION", "B-ANSWER", "B-HEADER", "I-ANSWER", "I-QUESTION", "I-HEADER"]
                        )
                    ),
                    "image": datasets.features.Image(),
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

       # 排序
        # TBYX
    def order_by_tbyx(self, bboxes, indexes):
        comb = [(bbox, index) for bbox, index in zip(bboxes, indexes)]
        sorted_comb = sorted(comb, key=lambda r: (r[0][1], [0][0]))
        for i in range(len(sorted_comb) - 1):
            for j in range(i, 0, -1):
                if abs(sorted_comb[j + 1][0][1] - sorted_comb[j][0][1]) < 20 and \
                        (sorted_comb[j + 1][0][0] < sorted_comb[j][0][0]):
                    tmp = sorted_comb[j]
                    sorted_comb[j] = sorted_comb[j + 1]
                    sorted_comb[j + 1] = tmp
                else:
                    break
        sorted_indexes = [index for _, index in sorted_comb]
        return sorted_indexes

    def _generate_examples(self, filepaths):
        #self.label2ids = XFUND_label2ids
        logger.info("Generating examples from = %s", filepaths)
        # open json
        with open(filepaths[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        # re-org data format
        total_data = {"id": [], "lines": [], "bboxes": [], "ner_tags": [], "image_path": []}
        for doc in data["documents"]:
            #image, _ = load_image(doc["img"]["fpath"])
            width, height = doc['img']['width'], doc['img']['height']
            doc["img"]["fpath"] = os.path.join(filepaths[1], doc["img"]["fname"])
            
            cur_doc_lines, cur_doc_bboxes, cur_doc_ner_tags, cur_doc_image_path = [], [], [], []
            for j in range(len(doc['document'])):
                # 每一行是一个分词
                cur_item = doc['document'][j]
                #cur_words = cur_item['text'] # Text是完整的句子，字符在words中
                cur_doc_lines.append(cur_item['text'])
                cur_doc_bboxes.append(self.box_norm(cur_item['box'], width=width, height=height))
                cur_doc_ner_tags.append(cur_item['label'])

            # 数据排序
            tbyx_index = self.order_by_tbyx(np.array(cur_doc_bboxes), np.arange(len(cur_doc_bboxes)))
            cur_doc_lines = np.array(cur_doc_lines)[tbyx_index]
            cur_doc_bboxes = np.array(cur_doc_bboxes)[tbyx_index]
            cur_doc_ner_tags = np.array(cur_doc_ner_tags)[tbyx_index]

            total_data['id'] += [len(total_data['id'])]
            total_data['lines'] += [cur_doc_lines]
            total_data['bboxes'] += [cur_doc_bboxes]
            total_data['ner_tags'] += [cur_doc_ner_tags]
            total_data['image_path'] += [doc["img"]["fpath"]]

        # Tokenize text and get bbox/label
        total_input_ids, total_bboxs, total_label_ids = [], [], []
        for i in range(len(total_data['lines'])):
            cur_doc_input_ids, cur_doc_bboxs, cur_doc_labels = [], [], []
            for j in range(len(total_data['lines'][i])):
                cur_input_ids = self.tokenizer(total_data['lines'][i][j], truncation=False, add_special_tokens=False, return_attention_mask=False)['input_ids']
                if len(cur_input_ids) == 0:
                    continue
                cur_label = total_data['ner_tags'][i][j].upper()
                if cur_label == 'OTHER':
                    cur_labels = ["O"] * len(cur_input_ids)
                    for k in range(len(cur_labels)):
                        cur_labels[k] = XFUND_label2ids[cur_labels[k]]
                else:
                    cur_labels = [cur_label] * len(cur_input_ids)
                    cur_labels[0] = XFUND_label2ids['B-' + cur_labels[0]]
                    for k in range(1, len(cur_labels)):
                        cur_labels[k] = XFUND_label2ids['I-' + cur_labels[k]]
                assert len(cur_input_ids) == len([total_data['bboxes'][i][j]] * len(cur_input_ids)) == len(cur_labels)
                cur_doc_input_ids += cur_input_ids
                cur_doc_bboxs += [total_data['bboxes'][i][j]] * len(cur_input_ids)
                cur_doc_labels += cur_labels
            assert len(cur_doc_input_ids) == len(cur_doc_bboxs) == len(cur_doc_labels)
            assert len(cur_doc_input_ids) > 0

            total_input_ids.append(cur_doc_input_ids)
            total_bboxs.append(cur_doc_bboxs)
            total_label_ids.append(cur_doc_labels)
        assert len(total_input_ids) == len(total_bboxs) == len(total_label_ids)

        # 数据分片
        input_ids, bboxs, labels = [], [], []
        image_path = []
        for i in range(len(total_input_ids)):
            start = 0
            cur_iter = 0
            while start < len(total_input_ids[i]):
                end = min(start + 512, len(total_input_ids[i]))

                input_ids.append(total_input_ids[i][start: end])
                bboxs.append(total_bboxs[i][start: end])
                labels.append(total_label_ids[i][start: end])

                image_path.append(total_data['image_path'][i])

                start = end
                cur_iter += 1

        assert len(input_ids) == len(bboxs) == len(labels)

        # 返回数据
        for i in range(len(input_ids)):
            image, _ = load_image(image_path[i])
            res = {
                "id": f"{i}",
                "input_ids": input_ids[i],
                "bboxes": bboxs[i],
                "labels": labels[i],
                "image": image,
            }
            yield i, res

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')