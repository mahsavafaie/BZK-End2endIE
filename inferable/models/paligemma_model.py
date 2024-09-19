from inferable.models.base_model import BaseModel
from inferable.models.utils import extract_info
from typing import Dict, Iterable, Tuple, List
import datasets
from PIL.Image import Image
import logging

import torch
import pytorch_lightning as pl
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
import re
from nltk import edit_distance
import numpy as np

import os
import datetime

logger = logging.getLogger(__name__)

# TODO: make everything really JSON instead of XML like structure
# TODO: change tokenizer such that has one token for each <s_BZK> and </s_BZK> element (similar to DONUT)
# TODO: check for peft model save and load in zero shot

# TODO: zero shot: just prompts: prompt = "What is the BZK number (usually denoted by A.Z. or Kartei-Nr.) of the document?"
class PaliGemmaZeroShot(BaseModel):
    def __init__(self, model_name: str = "google/paligemma-3b-pt-224", task_prompt: str = "extract JSON.", ordered_dataset_keys: list[str] = [], max_length = 512) -> None:
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

        model = PaliGemmaForConditionalGeneration.from_pretrained(self.model_name)
        processor = AutoProcessor.from_pretrained(self.model_name)

        for image in test_data:
            inputs = processor(text = self.task_prompt, images = image, return_tensors="pt")
            generated_ids = model.generate(**inputs, max_new_tokens=self.max_length)

            image_token_index = model.config.image_token_index
            num_image_tokens = len(generated_ids[generated_ids==image_token_index])
            num_text_tokens = len(processor.tokenizer.encode(self.task_prompt))
            num_prompt_tokens = num_image_tokens + num_text_tokens + 2
            generated_text = processor.batch_decode(generated_ids[:, num_prompt_tokens:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            return_dict = {}
            for dataset_key in self.ordered_dataset_keys:
                return_dict[dataset_key] = extract_info(generated_text, dataset_key, allow_partial_match=False)
            
            yield return_dict


### Begin of training code

class PaliGemmaDataset(torch.utils.data.Dataset):

    def __init__(self, dataset: datasets.arrow_dataset.Dataset, ordered_dataset_keys: List[str]):
        super().__init__()
        self.dataset = dataset
        self.dataset_length = len(self.dataset)
        self.ordered_dataset_keys = ordered_dataset_keys
    
    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.dataset[idx]
        target_sequence = ''
        #iterate over column header and value - it is important that the order is the same thus we use ordered_dataset_keys
        for column in self.ordered_dataset_keys:
            column_value = sample[column]
            if column_value is None:
                column_value = ''
            target_sequence += f"<s_{column}>{column_value}</s_{column}>"

        # remove attributes that are empty
        #for column in prediction_columns:
        #    if row[column]:
        #        target_sequence += f"<s_{column}>{row[column]}</s_{column}>"
        return {'image': sample["image"], 'target_sequence' : target_sequence}



class PalliGemmaPLModule(pl.LightningModule):
    def __init__(self, processor, model, learning_rate, max_length):
        super().__init__()
        self.processor = processor
        self.model = model
        self.my_learning_rate = learning_rate
        self.max_length = max_length

    def training_step(self, batch, batch_idx):
        outputs = self.model(input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                token_type_ids=batch['token_type_ids'],
                                pixel_values=batch['pixel_values'],
                                labels=batch['labels'])
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        #print("validation step start")
        input_ids, attention_mask, pixel_values, answers = batch

        #print("validation step generate")
        # autoregressively generate token IDs
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       pixel_values=pixel_values, max_new_tokens=self.max_length)
        #print("validation step generate end")
        # turn them back into text, chopping of the prompt
        # important: we don't skip special tokens here, because we want to see them in the output
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)
        #print("validation step write")
        scores = []
        with open('output/paligemma_validation', 'a') as f:
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
        #print("validation step end")
        return scores


    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.Adam(self.parameters(), lr=self.my_learning_rate)
        return optimizer

class PaliGemmaModel(BaseModel):
    
    def __init__(self) -> None:
        self.model_name = "google/paligemma-3b-pt-224"
        self.prompt = "extract JSON."
        self.finetuned_model_name = None
        self.processor = None
        self.max_length = 512
        
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
        self.best_model_path = None

        self.ordered_dataset_keys = []
    
    def train_collate_fn(self, examples):
        images = [example['image'] for example in examples]
        texts = [self.prompt for _ in range(len(images))]
        labels = [example['target_sequence'] for example in examples]

        inputs = self.processor(text=texts, images=images, suffix=labels, return_tensors="pt", padding=True,
                            truncation="only_second", max_length=self.max_length,
                            tokenize_newline_separately=False)
        return inputs

    def eval_collate_fn(self, examples):
        images = [example['image'] for example in examples]
        texts = [self.prompt for _ in range(len(images))]
        answers = [example['target_sequence'] for example in examples]

        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True, tokenize_newline_separately=False)

        # if returned as a dict, pytorch lightning will not be able to ignore the list (when moving all tensors to the device)
        #inputs['answers'] = answers
        #return inputs

        return inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"], answers

    def fit(self, training_data: datasets.arrow_dataset.Dataset, validation_dat: datasets.arrow_dataset.Dataset) -> None:
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        self.ordered_dataset_keys = list(training_data.features.keys())
        self.ordered_dataset_keys.remove('image')

        train_dataset = PaliGemmaDataset(training_data, self.ordered_dataset_keys)
        val_dataset = PaliGemmaDataset(validation_dat, self.ordered_dataset_keys)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=self.train_collate_fn, batch_size=self.batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, collate_fn=self.eval_collate_fn, batch_size=self.batch_size, shuffle=False)

        #x = next(iter(train_dataloader))
        #print(self.processor.batch_decode(x['input_ids']))

        #for id, label in zip(x['input_ids'][0][-30:], x['labels'][0][-30:]):
        #    print(self.processor.decode([id.item()]), self.processor.decode([label.item()]))

        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_type=torch.bfloat16
            ) if self.finetuning_quantized else None
        model = PaliGemmaForConditionalGeneration.from_pretrained(self.model_name, quantization_config=bnb_config, device_map="auto")

        if self.finetuning_lora:
            lora_config = LoraConfig(
                r=8,
                target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        else:
            for param in model.vision_tower.parameters():
                param.requires_grad = False

            for param in model.multi_modal_projector.parameters():
                param.requires_grad = False

        # model is created

        pl_model = PalliGemmaPLModule(self.processor, model, self.learning_rate, self.max_length)

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

        mydir = os.path.join("output/paligemma_model/", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.finetuned_model_name = mydir
        os.makedirs(mydir)
        pl_model.model.save_pretrained(mydir)
        pl_model.processor.save_pretrained(mydir)

    def predict(self, test_data: Iterable[Image]) -> Iterable[Dict[str, str]]:
        if self.finetuned_model_name is None or len(self.ordered_dataset_keys) == 0:
            raise ValueError("The model has not been trained yet. Please call the fit method first.")
        
        return PaliGemmaZeroShot(model_name=self.finetuned_model_name, 
                                  task_prompt=self.prompt, 
                                  ordered_dataset_keys=self.ordered_dataset_keys).predict(test_data)
    
    def __str__(self):
        return "PaliGemma"