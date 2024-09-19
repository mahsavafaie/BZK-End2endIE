from inferable.models.base_model import BaseModel
from inferable.models.utils import extract_info
from inferable.models.pytorch_utils import PyTorchDatasetWrapper
from typing import Dict, Iterable, Tuple, List
import datasets
from PIL.Image import Image
import logging

import torch
import pytorch_lightning as pl
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import re
from nltk import edit_distance
import numpy as np

import os
import datetime

logger = logging.getLogger(__name__)



class LLaVaNextZeroShot(BaseModel):
    def __init__(self, model_name: str = "", task_prompt: str = "Extract JSON", ordered_dataset_keys: list[str] = [], max_length = 512) -> None:
        self.model_name = model_name
        self.task_prompt = task_prompt
        self.ordered_dataset_keys = ordered_dataset_keys
        self.max_length = max_length

    def fit(self, training_data: datasets.arrow_dataset.Dataset, validation_dat: datasets.arrow_dataset.Dataset) -> None:
        self.ordered_dataset_keys = list(training_data.features.keys())
        self.ordered_dataset_keys.remove('image')

    def predict(self, test_data: Iterable[Image]) -> Iterable[Dict[str, str]]:
        if self.ordered_dataset_keys is None or len(self.ordered_dataset_keys) == 0:
            raise ValueError("The model has not been trained yet. Please call the fit method first.")

        model = LlavaNextForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch.float16) #TODO: check quantization
        processor = AutoProcessor.from_pretrained(self.model_name)

        for image in test_data:
            # prepare image and prompt for the model
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.task_prompt},
                    ],
                },
            ]
            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(text=text_prompt, images=[image], return_tensors="pt").to("cuda")
            generated_ids = model.generate(**inputs, max_new_tokens=self.max_length)

            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return_dict = {}
            for dataset_key in self.ordered_dataset_keys:
                return_dict[dataset_key] = extract_info(generated_text, dataset_key, allow_partial_match=False)
            
            yield return_dict


### Begin of training code

class LlavaModelPLModule(pl.LightningModule):
    def __init__(self, processor, model, learning_rate, max_length):
        super().__init__()
        self.processor = processor
        self.model = model
        self.my_learning_rate = learning_rate
        self.max_length = max_length

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, image_sizes, labels = batch
        outputs = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            image_sizes=image_sizes,
                            labels=labels
                          )
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        input_ids, attention_mask, pixel_values, image_sizes, answers = batch

        #print("validation step generate")
        # autoregressively generate token IDs
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       pixel_values=pixel_values, image_sizes=image_sizes, max_new_tokens=self.max_length)
        #print("validation step generate end")

        # turn them back into text, chopping of the prompt
        # important: we don't skip special tokens here, because we want to see them in the output
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

        scores = []
        with open('output/llava_validation', 'a') as f:
            for pred, answer in zip(predictions, answers):
                #                                                                |   |
                #                                                                v   v
                # remove potential space bewfore and after tag. example: <s_name> foo </s_name>
                pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
                computed_edit_distance = edit_distance(pred, answer)
                computed_normalized_edit_distance = computed_edit_distance / max(len(pred), len(answer))

                scores.append(computed_normalized_edit_distance)

                f.write(f"Prediction: {pred}\n")
                f.write(f"    Answer: {answer}\n")
                f.write(f" Normed ED: {computed_normalized_edit_distance}\n")
                f.write("\n")
            f.write("=========================================================\n")

        self.log("val_edit_distance", np.mean(scores))
        return scores


    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.Adam(self.parameters(), lr=self.my_learning_rate)
        return optimizer

class LLaVaNextModel(BaseModel):
    """LLaVa Next Model basic adoption of https://github.com/NielsRogge/Transformers-Tutorials/tree/master/LLaVa-NeXT"""
    
    def __init__(self) -> None:
        self.model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
        self.keep_empty_columns = True
        self.prompt = "Extract JSON"
        self.processor = None
        self.max_length = 512

        self.finetuned_model_name = None
        
        # following two boolean result in following combinations 'full', 'lora', 'quantized', or 'qlora'
        self.finetuning_lora = False
        self.finetuning_quantized = False

        #training related
        self.learning_rate = 1e-4
        self.batch_size = 2
        self.max_epochs = 25 # initial was 30
        self.accumulate_grad_batches = 8
        self.check_val_every_n_epoch = 30  # TODO: change back inital value is 1 to check every epoch (but this is too slow)
        self.gradient_clip_val = 1.0

        self.ordered_dataset_keys = []
    
    def train_collate_fn(self, examples):
        images = []
        texts = []

        for example in examples:
            image, ground_truth = example
            images.append(image)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ground_truth},
                    ],
                }
            ]
            text_prompt = self.processor.apply_chat_template(conversation)
            texts.append(text_prompt)

        batch = self.processor(text=texts, images=images, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pixel_values = batch["pixel_values"]
        image_sizes = batch["image_sizes"]
        labels = batch["labels"]

        return input_ids, attention_mask, pixel_values, image_sizes, labels

    def eval_collate_fn(self, examples):
        # We only feed the prompt to the model, so we don't add assistant's turn
        # Rather we indicate to `add_generation_prompt=True`

        images = []
        texts = []
        answers = []
        for example in examples:
            image, ground_truth = example
            images.append(image)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.prompt},
                    ],
                },
            ]
            text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            texts.append(text_prompt)
            answers.append(ground_truth)

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pixel_values = batch["pixel_values"]
        image_sizes = batch["image_sizes"]

        return input_ids, attention_mask, pixel_values, image_sizes, answers

    def find_all_linear_names_for_llava_next(self, model):
        cls = torch.nn.Linear
        lora_module_names = set()
        multimodal_keywords = ['multi_modal_projector', 'vision_model']
        for name, module in model.named_modules():
            if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                continue
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names: # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def fit(self, training_data: datasets.arrow_dataset.Dataset, validation_dat: datasets.arrow_dataset.Dataset) -> None:
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right side

        self.ordered_dataset_keys = list(training_data.features.keys())
        self.ordered_dataset_keys.remove('image')

        train_dataset = PyTorchDatasetWrapper(training_data, self.ordered_dataset_keys, self.keep_empty_columns)
        val_dataset = PyTorchDatasetWrapper(validation_dat, self.ordered_dataset_keys, self.keep_empty_columns)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=self.train_collate_fn, batch_size=self.batch_size, shuffle=True) # TODO:  num_workers=4 ?
        val_dataloader = torch.utils.data.DataLoader(val_dataset, collate_fn=self.eval_collate_fn, batch_size=self.batch_size, shuffle=False)

        #x = next(iter(train_dataloader))
        #print(self.processor.batch_decode(x['input_ids']))

        #for id, label in zip(x['input_ids'][0][-30:], x['labels'][0][-30:]):
        #    print(self.processor.decode([id.item()]), self.processor.decode([label.item()]))


        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_type=torch.float16
            ) if self.finetuning_quantized else None
        
        if self.finetuning_lora:
            model = LlavaNextForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch.float16, quantization_config=bnb_config) # TODO: device_map="auto" ?

            lora_config = LoraConfig(
                r=8,
                lora_alpha=8,
                lora_dropout=0.1,
                target_modules=self.find_all_linear_names_for_llava_next(model),
                init_lora_weights="gaussian", # TODO: task_type="CAUSAL_LM" ?
            )
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        else:
            # for full fine-tuning, we can speed up the model using Flash Attention
            # only available on certain devices, see https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
            model = LlavaNextForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch.float16, quantization_config=bnb_config,
                                                                      _attn_implementation="flash_attention_2",) # TODO: device_map="auto" ?

            # TODO: check which parts of the model to freeze
            #for param in model.vision_tower.parameters():
            #    param.requires_grad = False

            #for param in model.multi_modal_projector.parameters():
            #    param.requires_grad = False


        # model is created

        pl_model = LlavaModelPLModule(self.processor, model, self.learning_rate, self.max_length)

        early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")

        trainer = pl.Trainer(
            accelerator="gpu",
            max_epochs=self.max_epochs,
            accumulate_grad_batches=self.accumulate_grad_batches,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            gradient_clip_val=self.gradient_clip_val,
            precision="16-mixed",
            limit_val_batches=5,
            num_sanity_val_steps=0,
            callbacks=[early_stop_callback],
        )

        trainer.fit(pl_model, train_dataloader, val_dataloader)

        mydir = os.path.join("output/llava_next_model/", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.finetuned_model_name = mydir
        os.makedirs(mydir)
        pl_model.model.save_pretrained(mydir)
        pl_model.processor.save_pretrained(mydir)

    def predict(self, test_data: Iterable[Image]) -> Iterable[Dict[str, str]]:
        if self.finetuned_model_name is None or len(self.ordered_dataset_keys) == 0:
            raise ValueError("The model has not been trained yet. Please call the fit method first.")
        
        return LLaVaNextZeroShot(model_name=self.finetuned_model_name, 
                                  task_prompt=self.prompt, 
                                  ordered_dataset_keys=self.ordered_dataset_keys).predict(test_data)
    
    def __str__(self):
        return "LLaVaNext"