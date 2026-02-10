import base64
import copy
from PIL import Image
from utils.InternVL3 import *

SYS = """
You are an AI assistant designed to help us understand spatial relationship in 3D indoor scene and finish visual question answering.
"""

BASELINE_PROMPT = """
You will be given one or two images and a spatial reasoning reasoning questions.
Your goal is to answer the spatial related question correctly.

Directly output an answer from the answer choices provided below.
You can add some analysis in your response, but remember to format the end of your answer according to the rule.

Now, according to the following image, answer the question from provided choices:
Question: {question}
Answer Choice: {answer_choice}

Answer: 
"""


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def format_gpt_content(contents):
    formatted_content = []
    for c in contents:
        formatted_content.append({"type": "text", "text": c[0]})
        if len(c) == 2: # has image
            formatted_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(c[1])}",
                        "detail": "high",
                    },
                }
            )
    return formatted_content

def format_internvl3_content(contents, model_device=None):
    formatted_content = {
        "question": "",
        "num_patches_list": [],
    }
    pixel_values = []
    for c in contents:
        formatted_content["question"] += c[0]
        formatted_content["question"] += "\n"
        if len(c) == 2: # has image
            formatted_content["question"] += "<image>\n"
            img_tensor = load_image(c[1], max_num=12).to(torch.bfloat16)
            if model_device is not None:
                img_tensor = img_tensor.to(model_device)
            else:
                img_tensor = img_tensor.cuda()

            pixel_values.append(img_tensor)
            formatted_content["num_patches_list"].append(img_tensor.size(0))
    formatted_content["pixel_values"] = torch.cat(pixel_values, dim=0)
    return formatted_content

def format_spatial_vqa_prompt_answer_baseline(
    question: str,
    answer_choices: list,
    images: list,
) -> (str, list):
    """
    Format a ChatGPT prompt (with optional images) for a spatial VQA scenario.
    
    Args:
        question (str): The question to answer.
        answer_choices (list): The list of possible answer choices.
        images (list): A list of local file paths to images for the current view.
        
    Returns:
        (str, list):
            - A system prompt describing ChatGPT's overarching role & guidelines.
            - A list of pieces of content (text or (text, image)) for ChatGPT.
            The 'image' part is a Base64-encoded string.
    """
    
    # 1) System prompt describing the assistant’s overall role & rules
    sys_prompt = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
        "You must analyze any provided images or observations and answer the question.\n\n"
    )
    
    # 2) Build the content list: each element is text or (text, base64_image).
    content = []
    
    # a) Intro: mention current images (if any)
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append((f"\nImage 1 is your current egocentric view\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # b) Present the question and answer choices
    q_text = f"Question: {question}\n"
    ac_text = "Answer Choices:\n"
    for choice in answer_choices:
        ac_text += f"{choice}\n"
    content.append((q_text,))
    content.append((ac_text,))
    
    # e) Final instructions and the "Answer:" line
    instructions = (
        "Output the exact answer from the choices.\n"
        "Answer: "
    )
    content.append((instructions,))
    
    return sys_prompt, content


def format_spatial_vqa_prompt_answer_baseline_text_cot(
    question: str,
    answer_choices: list,
    images: list,
) -> (str, list):
    """
    Format a ChatGPT prompt (with optional images) for a spatial VQA scenario.
    
    Args:
        question (str): The question to answer.
        answer_choices (list): The list of possible answer choices.
        images (list): A list of local file paths to images for the current view.
        
    Returns:
        (str, list):
            - A system prompt describing ChatGPT's overarching role & guidelines.
            - A list of pieces of content (text or (text, image)) for ChatGPT.
            The 'image' part is a Base64-encoded string.
    """
    
    # 1) System prompt describing the assistant’s overall role & rules
    sys_prompt = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
        "You must analyze any provided images or observations and answer the question with step-by-step thinking.\n\n"
    )
    
    # 2) Build the content list: each element is text or (text, base64_image).
    content = []
    
    # a) Intro: mention current images (if any)
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append((f"\nImage 1 is your current egocentric view\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # b) Present the question and answer choices
    q_text = f"Question: {question}\n"
    ac_text = "Answer Choices:\n"
    for choice in answer_choices:
        ac_text += f"{choice}\n"
    content.append((q_text,))
    content.append((ac_text,))
    
    # e) Final instructions and the "Answer:" line
    instructions = (
        "Output the your step-by-step reasoning progresses and exact answer from the choices.\n"
        "Thoughts: "
        "Answer: "
    )
    content.append((instructions,))
    
    return sys_prompt, content


def ordered_to_action_consequences(
    ordered, 
    sampling_interval_angle: int,
    sampling_interval_meter: float,
):
    """
    ordered: List[(action_key, [img_paths])]
    return:
      {
        action_key: { "step-01 (9 deg)": path, ... }  # or meters
      }
    """
    action_consequences = {}

    for action_key, paths in ordered:
        # infer unit by action_key prefix
        if action_key.startswith("move-forward"):
            unit = "meters"
            interval = sampling_interval_meter
        else:
            unit = "degrees"
            interval = sampling_interval_angle

        sub = {}
        for i, p in enumerate(paths, start=1):
            # label could be simple:
            # sub[f"step-{i:02d}"] = p

            # or label with metric:
            metric_val = i * interval
            if unit == "meters":
                metric_str = f"{metric_val:.2f} m"
            else:
                metric_str = f"{int(metric_val)} deg"
            sub[f"step-{i:02d} ({metric_str})"] = p

        action_consequences[action_key] = sub

    return action_consequences


def format_spatial_vqa_prompt_answer_scaling(
    question: str,
    answer_choices: list,
    images: list,
    action_consequences: dict
) -> (str, list):
    """
    Format a ChatGPT prompt for a spatial VQA scenario in which we
    present multiple candidate actions *before* the assistant chooses one.
    
    Arguments:
        question (str): The question to answer.
        answer_choices (list): The list of possible answer choices.
        images (list): A list of local file paths to the current/initial view images.
        action_consequences (dict): A nested dictionary of candidate actions and their corresponding images.
            The structure is:
                {
                    "action_1": {
                        "subaction_1": "path_to_image",
                        "subaction_2": "path_to_image",
                        ...
                    },
                    "action_2": {
                        "subaction_1": "path_to_image",
                        "subaction_2": "path_to_image",
                        ...
                    },
                    ...
                }

    Returns:
        (str, list):
            - A system prompt describing ChatGPT's role & guidelines.
            - A list of pieces of content (text or (text, base64_image)) for ChatGPT.
              The 'image' part is a Base64-encoded string.
    """
    
    # ------------------ 1) System Prompt ------------------
    sys_prompt = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
        "You must analyze any provided images or observations and answer the question with step-by-step thinking.\n\n"
        "Rules:\n"
        "1. You should output the exact answer from the choices.\n"
        "2. You will be provided with multiple imagined views if you taking corresponding actions to help you answer the questions.\n"
        "3. Your final line must only include the exact answer choice.\n"
    )
    
    # Prepare the content list: text or (text, base64_image)
    content = []
    
    # ------------------ 2) Current Images ------------------
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append(("\nImage 1 is your current egocentric view\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # ------------------ 3) The Question & Choices ------------------
    q_text = f"Question: {question}\n\n"
    ac_text = "Answer Choices:\n"
    for choice in answer_choices:
        ac_text += f"  - {choice}\n"
    ac_text += "\n"
    
    content.append((q_text,))
    content.append((ac_text,))
    
    # ------------------ 4) Present Candidate Actions + Images ------------------
    # e.g. "turn-left 30" -> [3 images], "turn-right 30" -> [3 images], etc.
    actions_intro = (
        "Below are the imagined views you would obtain if you took the corresponding actions. "
        "These are provided to help you answer the question.\n"
        "You can include them in your reasoning, but you should still only output the exact answer at the last line\n"
    )
    content.append((actions_intro,))
    # import pdb; pdb.set_trace() 
    
    
    # old
    # for action_str, subaction_consequences in action_consequences.items():
    #     content.append((f"Action: {action_str}\n",))
    #     for subaction_str, img_path in subaction_consequences.items():
    #         content.append((f"{subaction_str}\n", img_path))
    #     content.append(("\n",))
        
    # new
    action_str = ''
    for idx, item in enumerate(action_consequences):
        action, img_path = item[0], item[1]
        if idx == 0:
            action_str += f"Action: {action}\n"
        else:
            action_str += f", then {action}\n"
            
        content.append((f"{action_str}\n", img_path))
        content.append(("\n",))
        
    # import pdb; pdb.set_trace()

    # ------------------ 5) Final Instructions + "Answer:" Prompt ------------------
    # The user can either pick an answer from the list or pick an action from the action_consequences.
    # instructions = (
    #     "Output the exact answer from the choices.\n"
    #     "Answer: "
    # )
    
    instructions = (
        "Output the your step-by-step reasoning progresses and exact answer from the choices.\n"
        "Thoughts: "
        "Answer: "
    )
    
    content.append((instructions,))
    
    return sys_prompt, content


def format_spatial_vqa_prompt_answer_scaling_no_cot(
    question: str,
    answer_choices: list,
    images: list,
    action_consequences: dict
) -> (str, list):
    """
    Format a ChatGPT prompt for a spatial VQA scenario in which we
    present multiple candidate actions *before* the assistant chooses one.
    
    Arguments:
        question (str): The question to answer.
        answer_choices (list): The list of possible answer choices.
        images (list): A list of local file paths to the current/initial view images.
        action_consequences (dict): A nested dictionary of candidate actions and their corresponding images.
            The structure is:
                {
                    "action_1": {
                        "subaction_1": "path_to_image",
                        "subaction_2": "path_to_image",
                        ...
                    },
                    "action_2": {
                        "subaction_1": "path_to_image",
                        "subaction_2": "path_to_image",
                        ...
                    },
                    ...
                }

    Returns:
        (str, list):
            - A system prompt describing ChatGPT's role & guidelines.
            - A list of pieces of content (text or (text, base64_image)) for ChatGPT.
              The 'image' part is a Base64-encoded string.
    """
    
    # ------------------ 1) System Prompt ------------------
    sys_prompt = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
        "You must analyze any provided images or observations and answer the question.\n\n"
        "Rules:\n"
        "1. You should output the exact answer from the choices.\n"
        "2. You will be provided with multiple imagined views if you taking corresponding actions to help you answer the questions.\n"
        "3. Your final line must only include the exact answer choice.\n"
    )
    
    # Prepare the content list: text or (text, base64_image)
    content = []
    
    # ------------------ 2) Current Images ------------------
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append(("\nImage 1 is your current egocentric view\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # ------------------ 3) The Question & Choices ------------------
    q_text = f"Question: {question}\n\n"
    ac_text = "Answer Choices:\n"
    for choice in answer_choices:
        ac_text += f"  - {choice}\n"
    ac_text += "\n"
    
    content.append((q_text,))
    content.append((ac_text,))
    
    # ------------------ 4) Present Candidate Actions + Images ------------------
    # e.g. "turn-left 30" -> [3 images], "turn-right 30" -> [3 images], etc.
    actions_intro = (
        "Below are the imagined views you would obtain if you took the corresponding actions. "
        "These are provided to help you answer the question.\n"
        "You can include them in your reasoning, but you should still only output the exact answer at the last line\n"
    )
    content.append((actions_intro,))
    # import pdb; pdb.set_trace() 
    
    # for action_str, subaction_consequences in action_consequences.items():
    #     content.append((f"Action: {action_str}\n",))
    #     for subaction_str, img_path in subaction_consequences.items():
    #         content.append((f"{subaction_str}\n", img_path))
    #     content.append(("\n",))
    
    action_str = ''
    # import pdb; pdb.set_trace()
    for idx, item in enumerate(action_consequences):
        action, img_path = item[0], item[1][0]
        if idx == 0:
            action_str += f"Action: {action}\n"
        else:
            action_str += f", then {action}\n"
            
        content.append((f"{action_str}\n", img_path))
        content.append(("\n",))
        
        
    # import pdb; pdb.set_trace() 
    
    # ------------------ 5) Final Instructions + "Answer:" Prompt ------------------
    # The user can either pick an answer from the list or pick an action from the action_consequences.
    # instructions = (
    #     "Output the exact answer from the choices.\n"
    #     "Answer: "
    # )
    
    instructions = (
        "Output the exact answer from the choices.\n"
        "Answer: "
    )
    
    content.append((instructions,))
    
    return sys_prompt, content



def format_spatial_vqa_prompt_answer_scaling_old(
    question: str,
    answer_choices: list,
    images: list,
    action_consequences: dict
) -> (str, list):
    """
    Format a ChatGPT prompt for a spatial VQA scenario in which we
    present multiple candidate actions *before* the assistant chooses one.
    
    Arguments:
        question (str): The question to answer.
        answer_choices (list): The list of possible answer choices.
        images (list): A list of local file paths to the current/initial view images.
        action_consequences (dict): A nested dictionary of candidate actions and their corresponding images.
            The structure is:
                {
                    "action_1": {
                        "subaction_1": "path_to_image",
                        "subaction_2": "path_to_image",
                        ...
                    },
                    "action_2": {
                        "subaction_1": "path_to_image",
                        "subaction_2": "path_to_image",
                        ...
                    },
                    ...
                }

    Returns:
        (str, list):
            - A system prompt describing ChatGPT's role & guidelines.
            - A list of pieces of content (text or (text, base64_image)) for ChatGPT.
              The 'image' part is a Base64-encoded string.
    """
    
    # ------------------ 1) System Prompt ------------------
    sys_prompt = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
        "You must analyze any provided images or observations and answer the question.\n\n"
        "Rules:\n"
        "1. You should output the exact answer from the choices.\n"
        "2. You will be provided with multiple imagined views if you taking corresponding actions to help you answer the questions.\n"
        "3. Your final line must only include the exact answer choice.\n"
    )
    
    # Prepare the content list: text or (text, base64_image)
    content = []
    
    # ------------------ 2) Current Images ------------------
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append(("\nImage 1 is your current egocentric view\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # ------------------ 3) The Question & Choices ------------------
    q_text = f"Question: {question}\n\n"
    ac_text = "Answer Choices:\n"
    for choice in answer_choices:
        ac_text += f"  - {choice}\n"
    ac_text += "\n"
    
    content.append((q_text,))
    content.append((ac_text,))
    
    # ------------------ 4) Present Candidate Actions + Images ------------------
    # e.g. "turn-left 30" -> [3 images], "turn-right 30" -> [3 images], etc.
    actions_intro = (
        "Below are the imagined views you would obtain if you took the corresponding actions. "
        "These are provided to help you answer the question.\n"
        "You can include them in your reasoning, but you should still only output the exact answer at the last line\n"
    )
    content.append((actions_intro,))
    # import pdb; pdb.set_trace() 
    
    for action_str, subaction_consequences in action_consequences.items():
        content.append((f"Action: {action_str}\n",))
        for subaction_str, img_path in subaction_consequences.items():
            content.append((f"{subaction_str}\n", img_path))
        content.append(("\n",))
    
    # action_str = ''
    # # import pdb; pdb.set_trace()
    # for idx, item in enumerate(action_consequences):
    #     action, img_path = item[0], item[1][0]
    #     if idx == 0:
    #         action_str += f"Action: {action}\n"
    #     else:
    #         action_str += f", then {action}\n"
            
    #     content.append((f"{action_str}\n", img_path))
    #     content.append(("\n",))
        
        
    # import pdb; pdb.set_trace() 
    
    # ------------------ 5) Final Instructions + "Answer:" Prompt ------------------
    # The user can either pick an answer from the list or pick an action from the action_consequences.
    # instructions = (
    #     "Output the exact answer from the choices.\n"
    #     "Answer: "
    # )
    
    instructions = (
        "Output the exact answer from the choices.\n"
        "Answer: "
    )
    
    content.append((instructions,))
    
    return sys_prompt, content

def format_spatial_vqa_prompt_answer_scaling_mmsi(
    question: str,
    images: list,
    action_consequences: dict
) -> (str, list):
    """
    Format a ChatGPT prompt for a spatial VQA scenario in which we
    present multiple candidate actions *before* the assistant chooses one.
    
    Arguments:
        question (str): The question to answer.
        answer_choices (list): The list of possible answer choices.
        images (list): A list of local file paths to the current/initial view images.
        action_consequences (dict): A nested dictionary of candidate actions and their corresponding images.
            The structure is:
                {
                    "action_1": {
                        "subaction_1": "path_to_image",
                        "subaction_2": "path_to_image",
                        ...
                    },
                    "action_2": {
                        "subaction_1": "path_to_image",
                        "subaction_2": "path_to_image",
                        ...
                    },
                    ...
                }

    Returns:
        (str, list):
            - A system prompt describing ChatGPT's role & guidelines.
            - A list of pieces of content (text or (text, base64_image)) for ChatGPT.
              The 'image' part is a Base64-encoded string.
    """
    
    # ------------------ 1) System Prompt ------------------
    sys_prompt = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
        "You must analyze any provided images or observations and answer the question with step-by-step thinking.\n\n"
        "Rules:\n"
        "1. You should output the exact answer from the choices.\n"
        "2. You will be provided with multiple imagined views (based on the 1st image) if you taking corresponding actions to help you answer the questions.\n"
        "3. Your final line must only include the exact answer choice.\n"
    )
    
    # Prepare the content list: text or (text, base64_image)
    content = []
    
    # ------------------ 2) Current Images ------------------
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
    else:
        content.append(("No image provided.\n\n",))
    
    # ------------------ 3) The Question & Choices ------------------
    q_text = f"Question: {question}\n\n"
    content.append((q_text,))
    
    # ------------------ 4) Present Candidate Actions + Images ------------------
    # e.g. "turn-left 30" -> [3 images], "turn-right 30" -> [3 images], etc.
    actions_intro = (
        "Below are the imagined views you would obtain if you took the corresponding actions. "
        "These are provided to help you answer the question.\n"
        "You can include them in your reasoning, but you should still only output the exact answer at the last line\n"
    )
    content.append((actions_intro,))
    # import pdb; pdb.set_trace() 
    
    for action_str, subaction_consequences in action_consequences.items():
        content.append((f"Action: {action_str}\n",))
        for subaction_str, img_path in subaction_consequences.items():
            content.append((f"{subaction_str}\n", img_path))
        content.append(("\n",))
        
    # import pdb; pdb.set_trace() 
    
    # ------------------ 5) Final Instructions + "Answer:" Prompt ------------------
    # The user can either pick an answer from the list or pick an action from the action_consequences.
    # instructions = (
    #     "Output the exact answer from the choices.\n"
    #     "Answer: "
    # )
    
    instructions = (
        "Output the your step-by-step reasoning progresses and answer with only option letter (e.g.: A) from the choices.\n"
        "Thoughts:\n"
        "Answer:"
    )
    
    content.append((instructions,))
    
    return sys_prompt, content

def format_spatial_vqa_prompt_answer_scaling(
    question: str,
    answer_choices: list,
    images: list,
    action_consequences: dict
) -> (str, list):
    """
    Format a ChatGPT prompt for a spatial VQA scenario in which we
    present multiple candidate actions *before* the assistant chooses one.
    
    Arguments:
        question (str): The question to answer.
        answer_choices (list): The list of possible answer choices.
        images (list): A list of local file paths to the current/initial view images.
        action_consequences (dict): A nested dictionary of candidate actions and their corresponding images.
            The structure is:
                {
                    "action_1": {
                        "subaction_1": "path_to_image",
                        "subaction_2": "path_to_image",
                        ...
                    },
                    "action_2": {
                        "subaction_1": "path_to_image",
                        "subaction_2": "path_to_image",
                        ...
                    },
                    ...
                }

    Returns:
        (str, list):
            - A system prompt describing ChatGPT's role & guidelines.
            - A list of pieces of content (text or (text, base64_image)) for ChatGPT.
              The 'image' part is a Base64-encoded string.
    """
    
    # ------------------ 1) System Prompt ------------------
    sys_prompt = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
        "You must analyze any provided images or observations and answer the question with step-by-step thinking.\n\n"
        "Rules:\n"
        "1. You should output the exact answer from the choices.\n"
        "2. You will be provided with multiple imagined views if you taking corresponding actions to help you answer the questions.\n"
        "3. Your final line must only include the exact answer choice.\n"
    )
    
    # Prepare the content list: text or (text, base64_image)
    content = []
    
    # ------------------ 2) Current Images ------------------
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append(("\nImage 1 is your current egocentric view\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # ------------------ 3) The Question & Choices ------------------
    q_text = f"Question: {question}\n\n"
    ac_text = "Answer Choices:\n"
    for choice in answer_choices:
        ac_text += f"  - {choice}\n"
    ac_text += "\n"
    
    content.append((q_text,))
    content.append((ac_text,))
    
    # ------------------ 4) Present Candidate Actions + Images ------------------
    # e.g. "turn-left 30" -> [3 images], "turn-right 30" -> [3 images], etc.
    actions_intro = (
        "Below are the imagined views you would obtain if you took the corresponding actions. "
        "These are provided to help you answer the question.\n"
        "You can include them in your reasoning, but you should still only output the exact answer at the last line\n"
    )
    content.append((actions_intro,))
    # import pdb; pdb.set_trace() 
    
    # for action_str, subaction_consequences in action_consequences.items():
    #     content.append((f"Action: {action_str}\n",))
    #     for subaction_str, img_path in subaction_consequences.items():
    #         content.append((f"{subaction_str}\n", img_path))
    #     content.append(("\n",))
    
    
    action_str = ''
    # import pdb; pdb.set_trace()
    for idx, item in enumerate(action_consequences):
        action, img_path = item[0], item[1][0]
        if idx == 0:
            action_str += f"Action: {action}\n"
        else:
            action_str += f", then {action}\n"
            
        content.append((f"{action_str}\n", img_path))
        content.append(("\n",))
        
    # import pdb; pdb.set_trace() 
    
    # ------------------ 5) Final Instructions + "Answer:" Prompt ------------------
    # The user can either pick an answer from the list or pick an action from the action_consequences.
    # instructions = (
    #     "Output the exact answer from the choices.\n"
    #     "Answer: "
    # )
    
    instructions = (
        "Output the your step-by-step reasoning progresses and exact answer from the choices.\n"
        "Thoughts: "
        "Answer: "
    )
    
    content.append((instructions,))
    
    return sys_prompt, content


def format_spatial_vqa_prompt_plan_scores(
    question: str,
    answer_choices: list,
    images: list,
    plan: list,
) -> (str, list):
    """
    Score each *plan* (trajectory) by how helpful its imagined views would be for answering the question.

    plan_candidates: List[dict], each dict:
      {
        "plan_id": str,
        "plan_str": str,               # human-readable plan
        "views": List[str]             # list of image paths (imagined views)
      }

    Output: a comma-separated list of integers 0-9, one per plan in the same order.
    """

    sys_prompt = (
        "You are an independent evaluator for spatial reasoning.\n"
        "You will be given a question, answer choices, the current observation image(s), and ONE candidate plan.\n"
        "The plan includes imagined views rendered by a world model.\n\n"

        "Your task is to SCORE THE PLAN by how useful its imagined views are for answering the question.\n"
        "Score range: 0 (not helpful / irrelevant / low quality) to 9 (highly helpful and informative).\n\n"

        "Scoring guidelines:\n"
        "- Higher score if the plan reveals missing evidence needed to answer (e.g., resolves occlusion or viewpoint ambiguity).\n"
        "- Higher score if the imagined views are sharp and coherent (not heavily distorted).\n"
        "- Lower score if the views do not change the evidence, are redundant, or are unrelated.\n"
        "- If the current images are already sufficient, most plans should receive low scores.\n\n"

        "Rules:\n"
        "1) Do NOT answer the question.\n"
        "2) Output ONLY a integer (0-9).\n"
        "3) Do not output any extra text.\n"
        "Output example:\n 5"
    )

    content = []
    content.append(("These are the images that pair with the question.\n",))
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append(("\nImage 1 is your current view.\n\n",))
    else:
        content.append(("No image provided.\n\n",))

    content.append((f"Question: {question}\n\n",))
    ac_text = "Answer Choices:\n"
    for choice in answer_choices:
        ac_text += f"  - {choice}\n"
    ac_text += "\n"
    content.append((ac_text,))

    content.append((
        "Below are candidate plans and their imagined views.\n"
        "Score each plan based on how helpful its views are for answering the question.\n\n",
    ))
    # import pdb; pdb.set_trace()
    # Present plans: plan header + multiple images
    # for i, p in enumerate(plan):
    #     # plan_id = plan[0].split('')[0] #.get("plan_id", f"plan_{i}")
    #     plan_str = p[0]#.get("plan_str", "")
    #     views = p[2] #.get("views", []) or []

    #     content.append((f"({plan_str}\n",))
    #     if len(views) == 0:
    #         content.append(("  (No imagined views)\n\n",))
    #         continue
    #     #for j, vpath in enumerate(views):
    #     content.append((f":", views))
    #     content.append(("\n",))
    
    action_str = ''
    for idx, item in enumerate(plan):
        action, img_path = item[0], item[1][0]
        if idx == 0:
            action_str += f"Action: {action}\n"
        else:
            action_str += f", then {action}\n"
            
        content.append((f"{action_str}\n", img_path))
        content.append(("\n",))
        
    instructions = (
        "Output a integer score.\n"
        "Output: "
    )
    content.append((instructions,))
    

    return sys_prompt, content


def format_spatial_vqa_prompt_scores(
    question: str,
    answer_choices: list,
    images: list,
    action_consequences: list,
    sys_prompt: str,
) -> (str, list):
    
    """
    Score the imaginations during the beam search process.
    """
    
    # ------------------ 1) System Prompt ------------------
    
    # Prepare the content list: text or (text, base64_image)
    content = []
    
    # ------------------ 2) Current Images ------------------
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append(("\nImage 1 is your current egocentric view\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # ------------------ 3) The Question & Choices ------------------
    q_text = f"Question: {question}\n\n"
    ac_text = "Answer Choices:\n"
    for choice in answer_choices:
        ac_text += f"  - {choice}\n"
    ac_text += "\n"
    
    content.append((q_text,))
    content.append((ac_text,))
    
    # ------------------ 4) Present Candidate Actions + Images ------------------
    # e.g. "turn-left 30" -> [3 images], "turn-right 30" -> [3 images], etc.

    action_intro = (
        f"Below are the imagined views after taking actions."
    )
    for index, action_consequence in enumerate(action_consequences):
        action_str, subaction_consequence, img_path = action_consequence
        content.append((action_intro,))
        content.append((f"Imagined image of index {str(index)} if you {subaction_consequence}:\n", img_path))
        content.append(("\n",))
    
    # ------------------ 5) Final Instructions + "Answer:" Prompt ------------------
    # The user can either pick an answer from the list or pick an action from the action_consequences.
    instructions = (
        "Output a list of scores.\n"
        "Output: "
    )
    content.append((instructions,))
    
    return sys_prompt, content

def format_spatial_vqa_prompt_answer_baseline_fill_in_blank(
    question: str,
    answer_choices: list,
    images: list = None,
) -> (str, list):
    """
    Format a ChatGPT prompt (with optional images) for a spatial VQA scenario.
    
    Args:
        question (str): The question to answer.
        answer_choices (list): The list of possible answer choices.
        images (list): A list of local file paths to images for the current view.
        
    Returns:
        (str, list):
            - A system prompt describing ChatGPT's overarching role & guidelines.
            - A list of pieces of content (text or (text, image)) for ChatGPT.
              The 'image' part is a Base64-encoded string.
    """
    
    # 1) System prompt describing the assistant’s overall role & rules
    sys_prompt = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
        "You must analyze any provided images or observations and answer the question.\n\n"
        "Rules:\n"
        "1. You should output the exact answer to fill in the blank, like directly output a floating-point number.\n"
        "2. Your final line must only include the exact answer choice.\n"
        "3. If there is an example format in the question, you should strictly follow it, otherwise you should only output a float-point number as the exact answer.\n"
        r"4. The final answer MUST BE put in \boxed{}."
    )
    
    # 2) Build the content list: each element is text or (text, base64_image).
    content = []
    
    # a) Intro: mention current images (if any)
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append(("\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # b) Present the question and answer choices
    q_text = f"Question: {question}\n"
    # ac_text = "Answer Choices:\n"
    # for choice in answer_choices:
    #     ac_text += f"{choice}\n"
    content.append((q_text,))
    # content.append((ac_text,))
    
    # e) Final instructions and the "Answer:" line
    instructions = (
        "Output the exact answer in a float-point number format.\n"
        "Answer: "
    )
    content.append((instructions,))
    
    return sys_prompt, content

def format_spatial_vqa_prompt_scores_fill_in_blank(
    # Currently hard code n=2
    question: str,
    answer_choices: list,
    images: list,
    action_consequences: list,
    sys_prompt: str,
) -> (str, list):
    
    """
    Score the views during the beam search process.
    """
    
    # ------------------ 1) System Prompt ------------------
    
    # Prepare the content list: text or (text, base64_image)
    content = []
    
    # ------------------ 2) Current Images ------------------
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append(("\nImage 1 is your current egocentric view\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # ------------------ 3) The Question & Choices ------------------
    q_text = f"Question: {question}\n\n"
    # ac_text = "Answer Choices:\n"
    # for choice in answer_choices:
    #     ac_text += f"  - {choice}\n"
    # ac_text += "\n"
    
    content.append((q_text,))
    # content.append((ac_text,))
    
    # ------------------ 4) Present Candidate Actions + Images ------------------
    # e.g. "turn-left 30" -> [3 images], "turn-right 30" -> [3 images], etc.

    action_intro = (
        f"Below are the imagined views after taking actions."
    )
    for index, action_consequence in enumerate(action_consequences):
        action_str, subaction_consequence, img_path = action_consequence
        content.append((action_intro,))
        content.append((f"Imagined image of index {str(index)} if you {subaction_consequence}:\n", img_path))
        content.append(("\n",))
    
    # ------------------ 5) Final Instructions + "Answer:" Prompt ------------------
    # The user can either pick an answer from the list or pick an action from the action_consequences.
    instructions = (
        "Output a list of scores.\n"
        "Output: "
    )
    content.append((instructions,))
    
    return sys_prompt, content

def format_spatial_vqa_prompt_rank(
    # Currently hard code n=2
    question: str,
    answer_choices: list,
    images: list,
    action_consequences: list,
) -> (str, list):
    
    """
    Rank the views during the beam search process.
    """
    
    # ------------------ 1) System Prompt ------------------
    sys_prompt = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
        "You must analyze any provided images and rank indexes of imagined images from most relevant to least relevant.\n\n"
        "Rules:\n"
        "1. You'll be provided with images (including imagined images), a question, and a set of answer choices. You should rank most relevant images that can help you answer the question from the choices.\n"
        "2. You should output a list of indexes, separated by ','. For example: Output: 3,1,2,0\n"
    )
    
    # Prepare the content list: text or (text, base64_image)
    content = []
    
    # ------------------ 2) Current Images ------------------
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            encoded_img = encode_image(img_path)
            content.append((f"Image {idx + 1}:", encoded_img))
        content.append(("\nImage 1 is your current egocentric view\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # ------------------ 3) The Question & Choices ------------------
    q_text = f"Question: {question}\n\n"
    ac_text = "Answer Choices:\n"
    for choice in answer_choices:
        ac_text += f"  - {choice}\n"
    ac_text += "\n"
    
    content.append((q_text,))
    content.append((ac_text,))
    
    # ------------------ 4) Present Candidate Actions + Images ------------------
    # e.g. "turn-left 30" -> [3 images], "turn-right 30" -> [3 images], etc.

    action_intro = (
        f"Below are the imagined views after taking actions."
    )
    for index, action_consequence in enumerate(action_consequences):
        action_str, subaction_consequence, img_path = action_consequence
        content.append((action_intro,))
        encoded_img = encode_image(img_path)
        content.append((f"Imagined image of index {str(index)} if you {subaction_consequence}:\n", encoded_img))
        content.append(("\n",))
    
    # ------------------ 5) Final Instructions + "Answer:" Prompt ------------------
    # The user can either pick an answer from the list or pick an action from the action_consequences.
    instructions = (
        "Output a list of indexes from most relevant image to least relevant image.\n"
        "Output: "
    )
    content.append((instructions,))
    
    return sys_prompt, content


def format_spatial_vqa_prompt_answer_scaling_fill_in_blank(
    question: str,
    answer_choices: list,
    images: list,
    action_consequences: dict
) -> (str, list):
    """
    Format a ChatGPT prompt for a spatial VQA scenario in which we
    present multiple candidate actions *before* the assistant chooses one.
    
    Arguments:
        question (str): The question to answer.
        answer_choices (list): The list of possible answer choices.
        images (list): A list of local file paths to the current/initial view images.
        action_consequences (dict): A nested dictionary of candidate actions and their corresponding images.
            The structure is:
                {
                    "action_1": {
                        "subaction_1": "path_to_image",
                        "subaction_2": "path_to_image",
                        ...
                    },
                    "action_2": {
                        "subaction_1": "path_to_image",
                        "subaction_2": "path_to_image",
                        ...
                    },
                    ...
                }

    Returns:
        (str, list):
            - A system prompt describing ChatGPT's role & guidelines.
            - A list of pieces of content (text or (text, base64_image)) for ChatGPT.
              The 'image' part is a Base64-encoded string.
    """
    
    # ------------------ 1) System Prompt ------------------
    sys_prompt = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
        "You must analyze any provided images or observations and answer the question.\n\n"
        "Rules:\n"
        "1. You should output the exact answer to fill in the blank, like directly output a floating-point number.\n"
        "2. You will be provided with multiple imagined views if you taking corresponding actions to help you answer the questions.\n"
        "3. You can include minimal reasoning, but your final line must only include the exact answer.\n"
        "4. If there is an example format in the question, you should strictly follow it, otherwise you should only output a float-point number as the exact answer.\n"
        r"5. The final answer MUST BE put in \boxed{}."
    )
    
    # Prepare the content list: text or (text, base64_image)
    content = []
    
    # ------------------ 2) Current Images ------------------
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append(("\nImage 1 is your current egocentric view\n",))
    else:
        content.append(("No image provided.\n\n",))
    
    # ------------------ 3) The Question & Choices ------------------
    q_text = f"Question: {question}\n\n"
    # ac_text = "Answer Choices:\n"
    # for choice in answer_choices:
    #     ac_text += f"  - {choice}\n"
    # ac_text += "\n"
    
    content.append((q_text,))
    # content.append((ac_text,))
    # content.append((ac_text,))
    
    # ------------------ 4) Present Candidate Actions + Images ------------------
    # e.g. "turn-left 30" -> [3 images], "turn-right 30" -> [3 images], etc.
    actions_intro = (
        "Below are the imagined views you would obtain if you took the corresponding actions.\n"
        "If there are more than one image in the question, these imaged views are based on the first image.\n"
        "These are provided to help you answer the question.\n"
        "You can include them in your reasoning, but you should still only output the exact answer at the last line\n"
    )
    content.append((actions_intro,))
    
    for action_str, subaction_consequences in action_consequences.items():
        content.append((f"Action: {action_str}\n",))
        for subaction_str, img_path in subaction_consequences.items():
            content.append((f"{subaction_str}\n", img_path))
        content.append(("\n",))
    
    # ------------------ 5) Final Instructions + "Answer:" Prompt ------------------
    # The user can either pick an answer from the list or pick an action from the action_consequences.
    instructions = (
        "Output the exact answer from the question.\n"
        "Answer: "
    )
    content.append((instructions,))
    
    return sys_prompt, content

def format_spatial_vqa_prompt_bbox(
    question: str,
    answer_choices: list,
    images: list,
) -> (str, list):
    sys_prompt = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D indoor scene. "
        "You must analyze the image and answer the question.\n\n"
        "Rules:\n"
        "1. Output the bounding box in your current egocentric view of the area most important and relevant for answering the question. For those questions containing marks, it is important to have the bounding box include the object that marked with the number mentioned in the question.\n"
        "2. The output should only contain two integer coordinates of the top-left and bottom-right corners of the bounding box, separated by ':' in the format (x1,y1):(x2,y2).\n"
        "3. Only output None if you are very uncertain about the bounding box location or it is not necessary for answering the question. This case is rare to happen.\n"
    )
    content = []
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        content.append((f"\nImage 1 is your current egocentric view of size {Image.open(images[0]).size}\n",))
    else:
        content.append(("No image provided.\n\n",))
    q_text = f"Question: {question}\n\n"
    ac_text = "Answer Choices:\n"
    for choice in answer_choices:
        ac_text += f"  - {choice}\n"
    ac_text += "\n"
    content.append((q_text,))
    content.append((ac_text,))
    instructions = (
        "Output either the bounding box coordinates in the format (x1,y1):(x2,y2) or None if uncertain or not needed.\n"
        "Output: "
    )
    content.append((instructions,))
    return sys_prompt, content

def format_spatial_vqa_prompt_policy_plan(
    question: str,
    answer_choices: list,
    images: list,
) -> (str, list):

    sys_prompt = (
            "You are a POLICY model for spatial reasoning in a 3D indoor scene. "
            "Your job is to decide whether to call a world model and, if so, plan actions that get the most useful imagined views.\n\n"

            "Input: image(s), a multiple-choice question, and answer options.\n\n"

            "Your tasks:\n"
            "1) Decide whether to SKIP or CALL the world model.\n"
            "2) If CALL, output a short action plan (1-6 actions) to get extra information.\n\n"

            "Action space (DISCRETE, fixed):\n"
            "- move-forward 0.25 meters\n"
            "- turn-left 9 degrees\n"
            "- turn-right 9 degrees\n"
            
            "Composing actions:\n"
            "- 2 turns ≈ 18°, 3 turns ≈ 27°, 5 turns ≈ 45°, 10 turns ≈ 90°.\n"
            "- When the question mentions a larger angle (e.g., 45°/90°), approximate it with repeated 9° turns.\n\n"

            "When to CALL the world model:\n"
            "- The answer is not directly visible from the current view.\n"
            "- The question depends on perspective, facing direction, rotation, or left/right relations.\n"
            "- There is ambiguity for motion between frames.\n\n"

            "Constraints:\n"
            "- No cancelling or oscillating actions (e.g., left then right).\n"
            "- If turning, choose ONE direction and turn monotonically.\n"

            "Output format (valid JSON only):\n"
            "{\n"
            '  \"decision\": \"skip\" or \"call_wm\",\n'
            '  \"reason\": \"<one sentence>\",\n'
            '  \"actions\": [\n'
            '    {\"type\": \"move-forward\"|\"turn-left\"|\"turn-right\", \"value\": <number>}\n'
            "  ]\n"
            "}\n"
        )
        



    content = []
    
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))

    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        try:
            content.append((f"\nImage 1 is your current view of size {Image.open(images[0]).size}\n",))
        except Exception:
            content.append(("\nImage 1 is your current view.\n",))
    else:
        content.append(("No image provided.\n\n",))

    q_text = f"Question: {question}\n\n"
    content.append((q_text,))

    if answer_choices:
        ac_text = "Answer Choices:\n"
        for choice in answer_choices:
            ac_text += f"  - {choice}\n"
        ac_text += "\n"
        content.append((ac_text,))

    instructions = (
        "Return JSON only following the required schema.\n"
        "Output: "
    )
    content.append((instructions,))
    return sys_prompt, content



def format_policy_plan_verification(
    question: str,
    answer_choices: list,
    images: list,
    policy_text: str,
) -> (str, list):

    
    sys_prompt = (
        "You are an independent verifier for a policy that decides whether to call a visual world model.\n"
        "Your job is to judge whether SKIPPING the world model is safe.\n\n"

        "IMPORTANT: The policy output (decision + reason) may be wrong or misleading.\n"
        "- Do NOT trust the policy's decision.\n"
        "- Do NOT reuse or paraphrase the policy's reason.\n"
        "- Base your judgment ONLY on the images + question + answer choices.\n\n"

        "Verification procedure (follow strictly):\n"
        "1) Identify what visual evidence is required to answer the question.\n"
        "2) Check whether that evidence is directly visible in the provided image(s).\n"
        "3) If any required evidence could be outside the field of view, occluded, behind an object, or viewpoint-dependent, you MUST set override=true.\n"
        "4) Set override=false ONLY if you are confident the correct answer is unambiguous from the current image(s) without any viewpoint change.\n\n"

        "Common cases where override=true:\n"
        "- The question mentions or implies viewpoint change: turn, move, after walking, next view, facing direction.\n"
        "- Left/right/front/behind/around relations that could flip with small rotations.\n"
        "- Occlusion: object might be behind another object, around a corner, or outside the current frame.\n"
        "- Multiple plausible spatial interpretations from the current view.\n\n"

        "Return JSON ONLY (no extra text). Your 'reason' must be based on missing/visible evidence, "
        "and must not reference the policy reason.\n"
        "{\n"
        '  "override": true/false,\n'
        '  "confidence": <number between 0 and 1>,\n'
        '  "reason": "<one short sentence about visible vs missing evidence>"\n'
        "}\n\n"

        "Sanity check before finalizing:\n"
        "- If you are uncertain, choose override=true.\n"
    )



    content = []
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))

    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
        try:
            content.append((f"\nImage 1 is your current egocentric view of size {Image.open(images[0]).size}\n",))
        except Exception:
            content.append(("\nImage 1 is your current egocentric view.\n",))
    else:
        content.append(("No image provided.\n\n",))

    q_text = f"Question: {question}\n\n"
    ac_text = "Answer Choices:\n"
    for choice in answer_choices:
        ac_text += f"  - {choice}\n"
    ac_text += "\n"
    content.append((q_text,))
    content.append((ac_text,))
    
    
    if policy_text is not None:
        content.append((f"Policy decision:\n{policy_text}\n\n",))

    instructions = (
        "Return JSON only following the required schema.\n"
        "Output: "
    )
    content.append((instructions,))
    return sys_prompt, content




def format_action_search(
    question: str,
    answer_choices: list,
    images: list,
    imagined_frames: list,
    # actions: list,
    action_consequences: dict
) -> (str, list):
    """
    Format a ChatGPT prompt (with optional images) for a spatial VQA scenario.
    
    Args:
        question (str): The question to answer.
        answer_choices (list): The list of possible answer choices.
        images (list): A list of local file paths to images for the current view.
        
    Returns:
        (str, list):
            - A system prompt describing ChatGPT's overarching role & guidelines.
            - A list of pieces of content (text or (text, image)) for ChatGPT.
            The 'image' part is a Base64-encoded string.
    """
    
    # 1) System prompt describing the assistant’s overall role & rules
    sys_prompt = (
    "You are an embodied reasoning agent navigating a 3D environment in order to answer a user's question.\n\n"
    "At every step, you are provided with:\n"
    "1. Action Space: {TURN_LEFT, TURN_RIGHT, MOVE_FORWARD}\n"
    "2. Current Frames: the agent's visual frame(s)\n"
    "3. Imagined Frames (optional): simulated views that may help with planning\n"
    "4. User Question: <QUESTION>\n\n"
    "Your role:\n"
    "- Analyze the current frames (and imagined frames if available).\n"
    "- Determine the most informative direction to explore next to eventually answer the question.\n"
    "- Use short, explicit reasoning to justify your decision.\n\n"
    "Output Format (JSON):\n"
    "{\n"
    "  \"reasoning\": \"<brief reasoning grounded in the observations>\",\n"
    "  \"next_action\": \"<one of: TURN_LEFT, TURN_RIGHT, MOVE_FORWARD>\"\n"
    "}\n\n"
    "Requirements:\n"
    "- Always output exactly one valid action from the action space.\n"
    "- Keep reasoning concise and grounded in the visual evidence.\n"
)

    # 2) Build the content list: each element is text or (text, base64_image).
    content = []
    
    # a) Intro: mention current images (if any)
    intro_text = "These are the images that pair with the question.\n"
    content.append((intro_text,))
    
    if images:
        for idx, img_path in enumerate(images):
            content.append((f"Image {idx + 1}:", img_path))
            
        content.append((f"\nImage 1 is your current real view\n",))
        content.append((f"\nIf provided, Image 2 is your next real view\n",))
    else:
        content.append(("No image provided.\n\n",))

    # c) Present the actions
    # action_intro = "These are the actions you have take.\n"
    # content.append((action_intro,))
    # if len(actions) > 0:
    #     for action in actions:
    #         content.append((f"{action}\n"))
    # else:
    #     content.append(("No action has been taken yet.\n\n",))

    if len(action_consequences) > 0:

        actions_intro = (
            "Below are the imagined views you would obtain if you took the corresponding actions. "
            "These are provided to help you answer the question.\n"
        )

        content.append((actions_intro,))

        for item in action_consequences:
            subaction_str, img_path = item[1], item[2]
            # content.append((f"Action: {action_str}\n",))
            # for subaction_str, img_path in subaction_consequences.items():
            content.append((f"Action: {subaction_str}\n", img_path))
            content.append(('then,\n',))

        content = content[:-1]
    # b) Present the question and answer choices
    q_text = f"Question: {question}\n"
    ac_text = "Answer Choices:\n"
    for choice in answer_choices:
        ac_text += f"{choice}\n"
    content.append((q_text,))
    content.append((ac_text,))
    

    instructions = (
        "Output **only valid JSON**:\n"
    )

    content.append((instructions,))
    
    return sys_prompt, content