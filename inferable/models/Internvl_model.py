from inferable.models.base_model import BaseModel
from typing import Dict, Iterable, Tuple, List
import datasets
from PIL.Image import Image
import logging

import torch
import math
import numpy as np
import torchvision.transforms as T
#from decord import VideoReader, cpu
from PIL import Image
import os
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

class InternvlModel(BaseModel):

    def __init__(self, model_name :str = "OpenGVLab/InternVL2-Llama3-76B") -> None:
        self.model_name = model_name
        self.predict_keys = None

    def fit(self, training_data: datasets.arrow_dataset.Dataset, validation_dat: datasets.arrow_dataset.Dataset) -> None:
        self.predict_keys = list(training_data.features.keys())
        self.predict_keys.remove('image')

    def predict(self, test_data: Iterable[Image]) -> Iterable[Dict[str, str]]:

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        def build_transform(input_size):
            MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
            return transform

        def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
            best_ratio_diff = float('inf')
            best_ratio = (1, 1)
            area = width * height
            for ratio in target_ratios:
                target_aspect_ratio = ratio[0] / ratio[1]
                ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_ratio = ratio
                elif ratio_diff == best_ratio_diff:
                    if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                        best_ratio = ratio
            return best_ratio

        def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
            orig_width, orig_height = image.size
            aspect_ratio = orig_width / orig_height

            # calculate the existing image aspect ratio
            target_ratios = set(
                (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
                i * j <= max_num and i * j >= min_num)
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

            # find the closest aspect ratio to the target
            target_aspect_ratio = find_closest_aspect_ratio(
                aspect_ratio, target_ratios, orig_width, orig_height, image_size)

            # calculate the target width and height
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
            blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

            # resize the image
            resized_img = image.resize((target_width, target_height))
            processed_images = []
            for i in range(blocks):
                box = (
                    (i % (target_width // image_size)) * image_size,
                    (i // (target_width // image_size)) * image_size,
                    ((i % (target_width // image_size)) + 1) * image_size,
                    ((i // (target_width // image_size)) + 1) * image_size
                )
                # split the image
                split_img = resized_img.crop(box)
                processed_images.append(split_img)
            assert len(processed_images) == blocks
            if use_thumbnail and len(processed_images) != 1:
                thumbnail_img = image.resize((image_size, image_size))
                processed_images.append(thumbnail_img)
            return processed_images

        def load_image(image, input_size=448, max_num=12):
            transform = build_transform(input_size=input_size)
            images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            return pixel_values

        def split_model(model_name):
            device_map = {}
            world_size = torch.cuda.device_count()
            num_layers = {
                'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
                'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
            # Since the first GPU will be used for ViT, treat it as half a GPU.
            num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
            num_layers_per_gpu = [num_layers_per_gpu] * world_size
            num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
            layer_cnt = 0
            for i, num_layer in enumerate(num_layers_per_gpu):
                for j in range(num_layer):
                    device_map[f'language_model.model.layers.{layer_cnt}'] = i
                    layer_cnt += 1
            device_map['vision_model'] = 0
            device_map['mlp1'] = 0
            device_map['language_model.model.tok_embeddings'] = 0
            device_map['language_model.model.embed_tokens'] = 0
            device_map['language_model.output'] = 0
            device_map['language_model.model.norm'] = 0
            device_map['language_model.lm_head'] = 0
            device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

            return device_map


        device_map = split_model('InternVL2-Llama3-76B')
        model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            #load_in_8bit=True,  #in 4-bit provides irrelevant results
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, use_fast=False)
        #load the model before the loop


        for image in test_data:
            #the inference code
            pixel_values = load_image(image, max_num=12).to(torch.bfloat16).cuda()
            generation_config = dict(max_new_tokens=1024, do_sample=False)

            def get_filename(image):
                if hasattr(image, "filename") and image.filename != "":
                    return os.path.basename(image.filename)
                return ""

            question = '''<image>\nPlease provide the following information as you can see on the image as a Python dictionary. If the information is not
            given, provide 'null' as the value for the key. BZK number is the code that from the top right of the image. : Compensation Office, BZK number, Applicant First Name,
            Applicant Last Name, Applicant Birth Name, Applicant Birthdate, Applicant Birth place, Applicant Address, Applicant's Marital Status, Victim First Name, Victim Last Name, Victim Birthdate,
            Victim Birth place, , Victim Death Date, Victim Death Place, Heirs'''
            response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
            print(f'file name: {get_filename(image)}\nAssistant: {response}')
            return_dict = {}
            #return_dict['Applicant First Name'] = response[:15]
            #return_dict['Applicant Last Name'] = response["Applicant Last Name"]

            yield return_dict

    def __str__(self):
        return "InternvlModel"
