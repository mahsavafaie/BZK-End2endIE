import importlib
from typing import List

models_map = {
    "Dummy": ("inferable.models.dummy_model", "DummyModel", {}),
    "Donut": ("inferable.models.donut_model", "DonutModel", {}),
    "DonutModelZeroShot": ("inferable.models.donut_model", "DonutModelZeroShot", {}),
    "PaliGemma": ("inferable.models.paligemma_model", "PaliGemmaModel", {}),
    "PaliGemmaZeroShot": ("inferable.models.paligemma_model", "PaliGemmaZeroShot", {}),
}


def __handle_arg_string(arg):
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg

def __simple_parse_args_string(args_string):
    """
    Parses something like
        args1=val1,arg2=val2
    Into a dictionary
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = [arg for arg in args_string.split(",") if arg]
    args_dict = {
        k: __handle_arg_string(v) for k, v in [arg.split("=") for arg in arg_list]
    }
    return args_dict

def get_models(model_texts: List[str]) -> List:
    if "all" in model_texts:
        model_texts = list(models_map.keys())
    models = []
    for model_text in model_texts:
        if model_text in models_map:
            module_name, class_name, default_arguments = models_map[model_text]
            my_class = getattr(importlib.import_module(module_name), class_name)
            models.append(my_class(**default_arguments))
        else:
            # default to parameters in command line like
            # --models class=LLM,model_name=meta-llama/Llama-2-70b-chat-hf,prompt=prompt1,revision=154f235,extractive_keywords_only=true
            model_args = __simple_parse_args_string(model_text)
            if "class" not in model_args:
                raise ValueError(f"Model argument {model_text} must contain a class key e.g. class=LLM")
            class_name = model_args.pop("class")
            if class_name not in models_map:
                raise ValueError(f"Model class {class_name} not found in models_map")
            module_name, class_name, default_arguments = models_map[class_name]
            arguments = {**default_arguments, **model_args}
            my_class = getattr(importlib.import_module(module_name), class_name)
            models.append(my_class(**arguments))
    return models