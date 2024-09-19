from inferable.models.base_model import BaseModel
from inferable.models.utils import extract_info
from typing import Dict, Iterable, Tuple, List
import datasets
from PIL.Image import Image
import logging

import torch
import pytorch_lightning as pl
from transformers import DonutProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
import re
from nltk import edit_distance
import numpy as np
import os
import datetime

logger = logging.getLogger(__name__)

# TODO: one model for each feature???
# TODO: check for batch sizes larger than one

class DonutModelZeroShot(BaseModel):
    def __init__(self, model_name :str = "naver-clova-ix/donut-base-finetuned-docvqa", task_prompt="<s_docvqa><s_question>{user_input}</s_question><s_answer>", ordered_dataset_keys: list[str] = []) -> None:
        self.model_name = model_name
        self.task_prompt = task_prompt
        self.ordered_dataset_keys = ordered_dataset_keys

    def fit(self, training_data: datasets.arrow_dataset.Dataset, validation_dat: datasets.arrow_dataset.Dataset) -> None:
        self.ordered_dataset_keys = list(training_data.features.keys())
        self.ordered_dataset_keys.remove('image')

    def predict(self, test_data: Iterable[Image]) -> Iterable[Dict[str, str]]:
        if self.ordered_dataset_keys is None or len(self.ordered_dataset_keys) == 0:
            raise ValueError("The model has not been trained yet. Please call the fit method first.")

        processor = DonutProcessor.from_pretrained(self.model_name)
        model = VisionEncoderDecoderModel.from_pretrained(self.model_name)

        #print("model.config.decoder_start_token_id: " + str(model.config.decoder_start_token_id))
        #print("model.config.decoder_start_token_id decode: " + processor.tokenizer.decode([model.config.decoder_start_token_id]))

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model.eval()
        model.to(device)

        # prepare decoder inputs
        decoder_input_ids = processor.tokenizer(self.task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        decoder_input_ids = decoder_input_ids.to(device)
        for image in test_data:
            pixel_values = processor(image, return_tensors="pt").pixel_values

            outputs = model.generate(
                pixel_values.to(device),
                decoder_input_ids=decoder_input_ids,
                max_length=model.decoder.config.max_position_embeddings,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

            sequence = processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
            sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
            #predicted_metadata = processor.token2json(sequence)
            #print(sequence)

            return_dict = {}
            for dataset_key in self.ordered_dataset_keys:
                return_dict[dataset_key] = extract_info(sequence, dataset_key, allow_partial_match=False)

            #print(predicted_metadata)
            #yield predicted_metadata
            yield return_dict


    def __str__(self):
        return "DonutModelZeroShot_" + self.model_name

### Begin of training code

# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Donut/CORD/Fine_tune_Donut_on_a_custom_dataset_(CORD)_with_PyTorch_Lightning.ipynb

class DonutDataset(torch.utils.data.Dataset):

    def __init__(self, dataset: datasets.arrow_dataset.Dataset, ordered_dataset_keys: List[str], processor: DonutProcessor,  is_train: bool, max_length: int, ignore_id: int = -100):
        super().__init__()
        self.dataset = dataset
        self.dataset_length = len(self.dataset)
        self.ordered_dataset_keys = ordered_dataset_keys
        self.processor = processor
        self.is_train = is_train
        self.max_length = max_length
        self.ignore_id = ignore_id

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.dataset[idx]

        # inputs (image data)
        pixel_values = self.processor(sample['image'], random_padding=self.is_train, return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze()

        # targets (text data)
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

        target_sequence += self.processor.tokenizer.eos_token # add the end of sequence token at the end

        input_ids = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length", # TODO: check max_length
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id # model doesn't need to predict pad token

        return {'pixel_values': pixel_values, 'labels': labels, 'target_sequence' : target_sequence}
        #return pixel_values, labels, target_sequence


class DonutPLModule(pl.LightningModule):
    def __init__(self, processor, model, learning_rate, max_length):
        super().__init__()
        self.processor = processor
        self.model = model
        self.donut_learning_rate = learning_rate
        self.max_length = max_length

    def training_step(self, batch, batch_idx):
        pixel_values, labels = batch['pixel_values'], batch['labels']

        outputs = self.model(pixel_values, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        pixel_values, answers = batch['pixel_values'], batch['target_sequence']
        batch_size = pixel_values.shape[0]

        # we feed the prompt to the model
        decoder_input_ids = torch.full((batch_size, 1), self.model.config.decoder_start_token_id, device=self.device)

        outputs = self.model.generate(pixel_values,
                                   decoder_input_ids=decoder_input_ids,
                                   max_length=self.max_length,
                                   #early_stopping=True, #  `num_beams` is set to 1. However, `early_stopping` is set to `True`
                                   pad_token_id=self.processor.tokenizer.pad_token_id,
                                   eos_token_id=self.processor.tokenizer.eos_token_id,
                                   use_cache=True,
                                   num_beams=1,
                                   bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                                   return_dict_in_generate=True,)

        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token <s_wieder>
            predictions.append(seq)

        scores = []
        with open('output/donut_validation_docvqa', 'a') as f:
            for pred, answer in zip(predictions, answers):
                #                                                                |   |
                #                                                                v   v
                # remove potential space bewfore and after tag. example: <s_name> foo </s_name>
                pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
                answer = answer.replace(self.processor.tokenizer.eos_token, "")
                computed_edit_distance = edit_distance(pred, answer)
                computed_normalized_edit_distance = computed_edit_distance / max(len(pred), len(answer))

                scores.append(computed_normalized_edit_distance)
                #print("lenscores" + str(len(scores)))
                #if len(scores) == 1:
                #    print(f"Prediction: {pred}")
                #    print(f"    Answer: {answer}")
                #    print(f" Normed ED: {computed_normalized_edit_distance}")
                f.write(predictions.index(pred))
                f.write(f"\nPrediction: {pred}\n")
                f.write(f"    Answer: {answer}\n")
                f.write(f" Normed ED: {computed_normalized_edit_distance}\n")
                f.write("\n")
            f.write("=========================================================\n")

        self.log("val_edit_distance", np.mean(scores))

        return scores

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.Adam(self.parameters(), lr=self.donut_learning_rate)

        return optimizer

class DonutModel(BaseModel):

    def __init__(self, model_name = "naver-clova-ix/donut-base") -> None:
        self.model_name = "naver-clova-ix/donut-base-finetuned-docvqa" # "naver-clova-ix/donut-base" # "naver-clova-ix/donut-base-finetuned-cord-v2"
        self.decoder_start_token = "<s_wieder>"
        self.finetuned_model_name = None
        self.image_size = [1280, 960] # (height, width)
        self.max_length = 768
        self.batch_size = 4 # feel free to increase the batch size if you have a lot of memory
        self.donut_learning_rate = 3e-5
        self.max_epochs = 50
        self.val_check_interval = None #when set to None at every Epoch all the data point in training data are loocked at first, before validation
        self.check_val_every_n_epoch = 2 #TODO: change back to 1
        self.gradient_clip_val = 1.0
        #self.best_model_path = None

        self.ordered_dataset_keys = []

    def setup_model(self, processor: DonutProcessor, model: VisionEncoderDecoderModel) -> None:

        # add tokens:
        list_of_tokens = [f"<s_{element_name}>" for element_name in self.ordered_dataset_keys] + [f"</s_{element_name}>" for element_name in self.ordered_dataset_keys]
        list_of_tokens.append(self.decoder_start_token)
        newly_added_num = processor.tokenizer.add_tokens(list_of_tokens)
        if newly_added_num > 0:
            model.decoder.resize_token_embeddings(len(processor.tokenizer))

        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([self.decoder_start_token])[0]

        # we update some settings which differ from pretraining; namely the size of the images + no rotation required
        # source: https://github.com/clovaai/donut/blob/master/config/train_cord.yaml
        processor.image_processor.size = self.image_size[::-1] # should be (width, height)
        processor.image_processor.do_align_long_axis = False


    def fit(self, training_data: datasets.arrow_dataset.Dataset, validation_dat: datasets.arrow_dataset.Dataset) -> None:

        # update image_size of the encoder
        # during pre-training, a larger image size was used
        config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
        config.encoder.image_size = self.image_size # (height, width)
        # update max_length of the decoder (for generation)
        config.decoder.max_length = self.max_length
        # TODO we should actually update max_position_embeddings and interpolate the pre-trained ones:
        # https://github.com/clovaai/donut/blob/0acc65a85d140852b8d9928565f0f6b2d98dc088/donut/model.py#L602

        processor = DonutProcessor.from_pretrained(self.model_name)
        model = VisionEncoderDecoderModel.from_pretrained(self.model_name, config=config)

        self.ordered_dataset_keys = list(training_data.features.keys())
        self.ordered_dataset_keys.remove('image')

        self.setup_model(processor, model)

        # create datasets
        train_dataset = DonutDataset(training_data, self.ordered_dataset_keys, processor, is_train=True, max_length=self.max_length)
        val_dataset = DonutDataset(validation_dat, self.ordered_dataset_keys, processor, is_train=False, max_length=self.max_length)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        # create the PyTorch Lightning module
        pl_model = DonutPLModule(processor, model,self.donut_learning_rate, self.max_length)

        early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_edit_distance", patience=5, verbose=False, mode="min")
        #checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_edit_distance', save_top_k=1)

        trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                max_epochs=self.max_epochs,
                val_check_interval=self.val_check_interval,
                check_val_every_n_epoch=self.check_val_every_n_epoch,
                gradient_clip_val=self.gradient_clip_val,
                precision=16, # we'll use mixed precision
                num_sanity_val_steps=0,
                #callbacks=[checkpoint_callback, early_stop_callback],
                callbacks=[early_stop_callback],
        )
        #https://github.com/Lightning-AI/pytorch-lightning/discussions/10399

        trainer.fit(pl_model, train_dataloader, val_dataloader)

        mydir = os.path.join("output/donut_model/", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.finetuned_model_name = mydir
        os.makedirs(mydir)
        pl_model.model.save_pretrained(mydir)
        pl_model.processor.save_pretrained(mydir)

        # TODO: implement a way to save the best model (created by ModelCheckpoint)

        # possible way to save the best model:
        #donut_checkpoint = torch.load("lightning_logs/version_8/checkpoints/epoch=19-step=1180.ckpt")
        #modified_state_dict = {k.replace("model.", ""): v for k, v in donut_checkpoint['state_dict'].items()}
        #model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name_or_path=None, state_dict=modified_state_dict, config=config, ignore_mismatched_sizes=True)
        #model.save_pretrained("output/donut_model/")
        # https://huggingface.co/docs/transformers/main_classes/model   from_pt=True

        #self.best_model_path = checkpoint_callback.best_model_path
        #logger.info(f"Best model path: {self.best_model_path}")


    def predict(self, test_data: Iterable[Image]) -> Iterable[Dict[str, str]]:
        if self.finetuned_model_name is None or len(self.finetuned_model_name) == 0:
            raise ValueError("The model has not been trained yet. Please call the fit method first.")

        return DonutModelZeroShot(model_name=self.finetuned_model_name,
                                  task_prompt=self.decoder_start_token,
                                  ordered_dataset_keys=self.ordered_dataset_keys).predict(test_data)

    def __str__(self):
        return "DonutModel"




# donut for docvqa
# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Donut/DocVQA/Batched_generation_with_Donut.ipynb
