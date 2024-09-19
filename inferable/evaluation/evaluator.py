from inferable.models.base_model import BaseModel
from inferable.data.base_dataset import BaseDataset
import os
import csv
from typing import List
import logging
import time
import datetime
from nltk import edit_distance
from collections import defaultdict
import xlsxwriter

logger = logging.getLogger(__name__)

def get_filename(image):
    if hasattr(image, "filename") and image.filename != "":
        return os.path.basename(image.filename)
    return ""

def format_numbers(number):
    return str(number).replace('.', ',') # f'{number:.15f}'.replace('.', ',') # .5

   

def evaluate(models : List[BaseModel], datasets : List[BaseDataset], output_folder : str):
    for dataset in datasets:
        dataset_name = dataset.__class__.__name__
        training_data, validation_data = dataset.get_training_data(), dataset.get_validation_data()
        test_data = dataset.get_test_data()
        for model in models:
            model_name = str(model)
            #process model name such that it can be used a file name
            model_name = model_name.replace("/", "_")

            # Train the model
            logger.info(f"Run training of model {model_name} on {dataset_name}")
            start_time = time.time()
            model.fit(training_data, validation_data)
            training_time = time.time() - start_time
            training_time_hours_minute_seconds = datetime.timedelta(seconds=training_time)
            logger.info(f"Finished training of model {model_name} on {dataset_name} in {training_time_hours_minute_seconds} (HH:MM:SS).")

            # Predict the metadata
            logger.info(f"Run prediction of model {model_name} on {dataset_name}")
            start_time = time.time()
            predicted_metadata = list(model.predict(test_data['image']))
            prediction_time = time.time() - start_time
            prediction_time_hours_minute_seconds = datetime.timedelta(seconds=prediction_time)
            logger.info(f"Finished training of model {model_name} on {dataset_name} in {prediction_time_hours_minute_seconds} (HH:MM:SS).")

            # Evaluate the predictions
            logger.info(f"Run evaluation of model {model_name} on {dataset_name}")
            start_time = time.time()
            # Calculate the evaluation metrics
            file_path = str(os.path.join(output_folder, f"evaluation_results-{model_name}-{dataset_name}"))
            with open(file_path + ".csv", 'w', newline='', encoding='utf-8') as csvfile: #,  xlsxwriter.Workbook(file_path + ".xlsx") as xlsxfile:
                writer = csv.writer(csvfile)

                predicted_keys = list(test_data.features.keys())
                predicted_keys.remove("image")
                
                header_row = ['image', 'avg_edit_distance', 'avg_normalized_edit_distance', 'avg_edit_distance_non_empty', 'avg_normalized_edit_distance_non_empty']
                for prediction_key in predicted_keys:
                    header_row.extend([f'{prediction_key}_ground_truth', f'{prediction_key}_predicted', f'{prediction_key}_edit_dist', f'{prediction_key}_norm_edit_dist'])
                header_row.extend(['training_time_seconds', 'prediction_time_seconds'])
                writer.writerow(header_row)
                #xlsxfile.write_row(0, 0,  header_row)

                edit_distance_dict = defaultdict(int)
                normalized_edit_distance_dict = defaultdict(float)
                count_of_non_empty_comparisons = defaultdict(int)
                for i, test_example in enumerate(test_data):
                    row = [get_filename(test_example['image'])]
                    predicted = predicted_metadata[i]
                    avg_edit_distance = 0
                    avg_normalized_edit_distance = 0
                    non_empty_comparisons_of_row = 0
                    for prediction_key in predicted_keys:
                        ground_truth_value = test_example[prediction_key]
                        predicted_value = predicted.get(prediction_key) # assign None if key does not exist

                        # TODO: check if still needed
                        if ground_truth_value is None:
                            ground_truth_value = ""
                        if predicted_value is None:
                            predicted_value = ""

                        if ground_truth_value != "" or predicted_value != "": # Only count non-empty comparisons -> if both are empty, we do not count it
                            count_of_non_empty_comparisons[prediction_key] += 1
                            count_of_non_empty_comparisons['all'] += 1
                            non_empty_comparisons_of_row += 1

                        #print(f"ground_truth_value: {ground_truth_value}")
                        #print(f"predicted_value: {predicted_value}")

                        computed_edit_distance = edit_distance(predicted_value, ground_truth_value)
                        maximum_length = max(len(predicted_value), len(ground_truth_value))
                        computed_normalized_edit_distance = computed_edit_distance / maximum_length if maximum_length > 0 else 0

                        avg_edit_distance += computed_edit_distance
                        avg_normalized_edit_distance += computed_normalized_edit_distance

                        edit_distance_dict[prediction_key] += computed_edit_distance
                        edit_distance_dict['all'] += computed_edit_distance

                        normalized_edit_distance_dict[prediction_key] += computed_normalized_edit_distance
                        normalized_edit_distance_dict['all'] += computed_normalized_edit_distance

                        row.extend([ground_truth_value, predicted_value, computed_edit_distance, format_numbers(computed_normalized_edit_distance)])

                    row.insert(1, format_numbers(avg_edit_distance / len(predicted_keys)))
                    row.insert(2, format_numbers(avg_normalized_edit_distance / len(predicted_keys)))
                    row.insert(3, format_numbers(avg_edit_distance / non_empty_comparisons_of_row))
                    row.insert(4, format_numbers(avg_normalized_edit_distance / non_empty_comparisons_of_row))

                    writer.writerow(row)
                    #xlsxfile.write_row(i + 1, 0,  row)
                
                test_length = len(test_data)
                avg_row = ['average_all_images', 
                           format_numbers(edit_distance_dict['all'] / (test_length*len(predicted_keys))), 
                           format_numbers(normalized_edit_distance_dict['all'] / (test_length*len(predicted_keys))), 
                           '', ''
                           ]
                for prediction_key in predicted_keys:
                    avg_row.extend(['', '',
                                    format_numbers(edit_distance_dict[prediction_key] / test_length), 
                                    format_numbers(normalized_edit_distance_dict[prediction_key] / test_length)])
                avg_row.extend([training_time, prediction_time])
                writer.writerow(avg_row)
                
                # just in case someone asks why the average of the averages is not the same as the average of all values
                # https://math.stackexchange.com/questions/95909/why-is-an-average-of-an-average-usually-incorrect
                avg_row = ['average_non_empty_comparisons', '', '', 
                           format_numbers(edit_distance_dict['all'] / count_of_non_empty_comparisons['all'] if count_of_non_empty_comparisons['all'] > 0 else 0), 
                           format_numbers(normalized_edit_distance_dict['all'] / count_of_non_empty_comparisons['all'] if count_of_non_empty_comparisons['all'] > 0 else 0)
                           ]
                for prediction_key in predicted_keys:
                    avg_row.extend(['', '', 
                                    format_numbers(edit_distance_dict[prediction_key] / count_of_non_empty_comparisons[prediction_key] if count_of_non_empty_comparisons[prediction_key] > 0 else 0), 
                                    format_numbers(normalized_edit_distance_dict[prediction_key] / count_of_non_empty_comparisons[prediction_key] if count_of_non_empty_comparisons[prediction_key] > 0 else 0)])
                avg_row.extend([training_time, prediction_time])
                writer.writerow(avg_row)

                
                # TODO: add images to the xlsx file
                # adding images to comments is not supported  https://www.youtube.com/watch?v=pPekR2rzwWI
                # https://github.com/jmcnamara/XlsxWriter/issues/823
                # https://stackoverflow.com/questions/69329336/add-image-into-comment-in-excel
            evaluation_time = time.time() - start_time
            evaluation_time_hours_minute_seconds = datetime.timedelta(seconds=evaluation_time)
            logger.info(f"Finished evaluation of model {model_name} on {dataset_name} in {evaluation_time_hours_minute_seconds} (HH:MM:SS).")