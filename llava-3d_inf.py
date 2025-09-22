"""Batch inference script for LLaVA-3D multi-view image evaluation.

Adapted from original PointLLM version to work with LLaVA-3D codebase.
This version handles multi-view RGB images instead of point clouds.
"""

import argparse
from transformers import AutoTokenizer
import torch
import os
import glob
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_special_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
import numpy as np
import json
from PIL import Image

# Suppress transformers logging errors
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)

class FakeUpload:
    """
    A simple container class to mimic file upload objects for video/scene inference.
    Adapted from original PointLLM gradio code to work with video directories.
    """
    def __init__(self, video_dir_path, hex, scene_name):
        self.name = video_dir_path  # Directory containing video/scene data
        self.hex = hex
        self.scene_name = scene_name

def make_named_upd_txt_files(identifier_at_scene_list, updtext_versionfolder_subfolder_path):
    """
    Returns a list of file paths for upd text samples based on the provided subfolder and scenes.
    """
    return [os.path.join(updtext_versionfolder_subfolder_path, name + ".txt") for name in identifier_at_scene_list]

def make_named_video_dirs(identifier_at_scene_list, video_path):
    """
    Returns a list of FakeUpload objects, each representing a video/scene directory.
    Handles multiple path structures:
    1. 3D-FRONT: video_path/identifier/scene/scene/renders_scene_*/
    2. Crops3D (new): video_path/identifier/scene/renders_scene_*/
    3. Crops3D (old): video_path/identifier/scene/scene/
    """
    results = []
    for identifier_at_scene in identifier_at_scene_list:
        identifier, scene = identifier_at_scene.split('@')
        
        # Try multiple path structures in order of preference
        renders_dir = None
        
        # Structure 1: Try new Crops3D format - video_path/identifier/scene/renders_scene_*/
        base_scene_dir = os.path.join(video_path, identifier, scene)
        renders_pattern = f"renders_{scene}_*"
        renders_dirs = glob.glob(os.path.join(base_scene_dir, renders_pattern))
        
        if renders_dirs:
            renders_dir = renders_dirs[0]
            print(f"[DEBUG] Found renders dir (new Crops3D format): {renders_dir}", flush=True)
        else:
            # Structure 2: Try 3D-FRONT format - video_path/identifier/scene/scene/renders_scene_*/
            nested_scene_dir = os.path.join(video_path, identifier, scene, scene)
            renders_dirs = glob.glob(os.path.join(nested_scene_dir, renders_pattern))
            
            if renders_dirs:
                renders_dir = renders_dirs[0]
                print(f"[DEBUG] Found renders dir (3D-FRONT format): {renders_dir}", flush=True)
            else:
                # Structure 3: Try old Crops3D format - video_path/identifier/scene/scene/
                if os.path.exists(nested_scene_dir):
                    renders_dir = nested_scene_dir
                    print(f"[DEBUG] Using nested scene dir (old Crops3D format): {renders_dir}", flush=True)
                else:
                    # Final fallback: use base scene directory
                    renders_dir = base_scene_dir
                    print(f"[WARN] No renders dir found for {scene}, using fallback: {renders_dir}", flush=True)
        
        results.append(FakeUpload(renders_dir, identifier, scene))
    return results

def load_multi_view_images(image_dir_path, max_images=6):
    """Load multiple view images from a directory for model input."""
    # This function is now deprecated since we use video processing
    # Keeping for backward compatibility but not used in main inference
    try:
        # Look for jpg files (RGB images)
        image_files = glob.glob(os.path.join(image_dir_path, "*.jpg"))
        image_files.sort()
        
        if not image_files:
            print(f"[ERROR] No .jpg images found in {image_dir_path}", flush=True)
            return None
            
        # Sample evenly spaced images up to max_images
        if len(image_files) > max_images:
            step = len(image_files) // max_images
            image_files = image_files[::step][:max_images]
        
        images = []
        for img_file in image_files:
            image = Image.open(img_file).convert("RGB")
            images.append(image)
            
        return images
    except Exception as e:
        print(f"[ERROR] Failed to load images from {image_dir_path}: {e}", flush=True)
        return None

def inference(
    scene_list_txt_file_path,
    updtext_versionfolder_subfolder_path,
    video_path,
    upd_subset_name,
    model,
    tokenizer,
    processor,
    keywords,
    mm_use_im_start_end,
    conv_template,
    json_tag=None,
):
    """Perform batch inference on video scenes and prompts."""
    with open(scene_list_txt_file_path, 'r') as f:
        identifier_at_scene_list = f.read().splitlines()
    scene_list_txt_filename_noext = os.path.basename(os.path.normpath(scene_list_txt_file_path)).replace('.txt', '')

    video_dir_list = make_named_video_dirs(identifier_at_scene_list, video_path)
    upd_txt_file_list = make_named_upd_txt_files(identifier_at_scene_list, updtext_versionfolder_subfolder_path)

    print(f"[DEBUG] Video path base: {video_path}", flush=True)
    print(f"[DEBUG] Number of scenes to process: {len(video_dir_list)}", flush=True)

    results = {}  # Dictionary to store all results
    
    total_samples = len(video_dir_list)
    for idx, (video_dir, txt_file) in enumerate(zip(video_dir_list, upd_txt_file_list), 1):
        try:
            print(f"[PROGRESS] Processing sample {idx}/{total_samples}: {video_dir.hex}@{video_dir.scene_name}", flush=True)
            # Read prompt
            with open(txt_file, 'r') as f:
                prompt = f.read().strip()

            # Load multi-view images directly from our 3D-FRONT renders directory
            render_dir = video_dir.name
            print(f"[DEBUG] Processing video path: {render_dir}", flush=True)
            print(f"[DEBUG] Video path exists: {os.path.exists(render_dir)}", flush=True)
            print(f"[DEBUG] Video path is absolute: {os.path.isabs(render_dir)}", flush=True)
            print(f"[DEBUG] Current working directory: {os.getcwd()}", flush=True)
            
            # Load images directly since our data doesn't match expected video processor formats
            image_files = glob.glob(os.path.join(render_dir, "*.png"))
            if not image_files:
                print(f"[ERROR] No PNG files found in: {render_dir}", flush=True)
                continue
                
            image_files.sort()  # Sort to ensure consistent ordering
            print(f"[DEBUG] Found {len(image_files)} images in render directory", flush=True)
            
            # Load images
            images = []
            for img_file in image_files[:6]:  # Limit to 6 images like demo
                try:
                    img = Image.open(img_file).convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"[WARN] Failed to load image {img_file}: {e}", flush=True)
                    
            if not images:
                print(f"[ERROR] No valid images loaded from: {render_dir}", flush=True)
                continue
                
            print(f"[DEBUG] Loaded {len(images)} images for processing", flush=True)
            
            # Process images using the image processor instead of video processor
            from llava.mm_utils import process_images
            
            # Get image sizes like the reference implementation
            image_sizes = [img.size for img in images]
            print(f"[DEBUG] Image sizes: {image_sizes}", flush=True)
            
            images_tensor = process_images(
                images,
                processor['image'],
                model.config
            ).to(model.device, dtype=torch.float16)
            
            # For 3D-FRONT data, we don't have depth/pose/intrinsics, so set them to None
            depths_tensor = None
            poses_tensor = None
            intrinsics_tensor = None
            clicks_tensor = None  # Set to None like reference implementation

            # Reset conversation and format prompt
            conv = conv_template.copy()
            
            # Add image tokens to the prompt
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in prompt:
                if mm_use_im_start_end:
                    prompt = prompt.replace(IMAGE_PLACEHOLDER, image_token_se)
                else:
                    prompt = prompt.replace(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN)
            else:
                if mm_use_im_start_end:
                    prompt = image_token_se + "\n" + prompt
                else:
                    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

            # Add to conversation
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt_formatted = conv.get_prompt()

            # Tokenize
            input_ids = tokenizer_special_token(prompt_formatted, tokenizer, return_tensors="pt").unsqueeze(0).cuda()
            print(f"[DEBUG] Input ids shape: {input_ids.shape}", flush=True)
            print(f"[DEBUG] Images tensor shape: {images_tensor.shape}", flush=True)

            # Set up stopping criteria
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            stop_str = keywords[0]
            print(f"[DEBUG] Stop string: '{stop_str}'", flush=True)

            # Generate response
            print(f"[DEBUG] Starting generation...", flush=True)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    depths=depths_tensor,
                    poses=poses_tensor,
                    intrinsics=intrinsics_tensor,
                    clicks=clicks_tensor,
                    image_sizes=None,  # Set to None like reference implementation
                    do_sample=True if 0.2 > 0 else False,  # Use temperature value
                    temperature=0.2,
                    max_new_tokens=512,  # Use same as reference
                    use_cache=True,
                    # stopping_criteria=[stopping_criteria]  # Temporarily removed
                )
            print(f"[DEBUG] Generation completed. Output ids shape: {output_ids.shape}", flush=True)

            # Decode response - decode all tokens like reference implementation
            print(f"[DEBUG] Input token length: {input_ids.shape[1]}", flush=True)
            print(f"[DEBUG] Generated token length: {output_ids.shape[1]}", flush=True)
            
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            print(f"[DEBUG] Raw output before processing: '{outputs}'", flush=True)
            
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            
            print(f"[DEBUG] Final processed output: '{outputs}'", flush=True)

            # Store result
            results[video_dir.hex + '@' + video_dir.scene_name.split(".")[0]] = {
                "prompt": prompt, 
                "response": outputs
            }

            print(f"[INFO] Processed {video_dir.hex}@{video_dir.scene_name}: {outputs}", flush=True)

        except Exception as e:
            print(f"[ERROR] Failed to process pair ({txt_file}, {video_dir.name}): {e}", flush=True)

    # Write all results to a JSON file
    try:
        tag_part = f"{json_tag}_" if json_tag else ""
        json_filename = f'inf_rslts_llava3d_{tag_part}{scene_list_txt_filename_noext}_{upd_subset_name}.json'
        with open(json_filename, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"[INFO] Results saved to {json_filename}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to write results to JSON file: {e}", flush=True)

def init_model(args):
    """Initialize the LLaVA-3D model for batch inference."""
    disable_torch_init()
    model_path = os.path.expanduser(args.model_name)
    print(f'[INFO] Loading model: {model_path}', flush=True)

    # Use LLaVA-3D model loading
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path, 
        args.model_base, 
        model_name, 
        torch_dtype=torch.float16
    )

    model.eval()

    # Determine conversation mode - Force llava_v1 for LLaVA-3D
    conv_mode = "llava_v1"
    
    # Original logic as backup:
    # if "llama-2" in model_name.lower():
    #     conv_mode = "llava_llama_2"
    # elif "mistral" in model_name.lower():
    #     conv_mode = "mistral_instruct"
    # elif "v1.6-34b" in model_name.lower():
    #     conv_mode = "chatml_direct"
    # elif "v1" in model_name.lower():
    #     conv_mode = "llava_v1"
    # elif "3D" in model_name.lower():
    #     conv_mode = "llava_v1"
    # elif "mpt" in model_name.lower():
    #     conv_mode = "mpt"
    # else:
    #     conv_mode = "llava_v0"

    print(f'[INFO] Using conversation mode: {conv_mode}', flush=True)
    
    mm_use_im_start_end = getattr(model.config, 'mm_use_im_start_end', False)
    
    conv = conv_templates[conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    
    return model, tokenizer, processor, keywords, mm_use_im_start_end, conv

def existing_dir(path):
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"readable_dir: '{path}' is not a valid directory")
    return path

def existing_file(path):
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"readable_file: '{path}' is not a valid file")
    return path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference for LLaVA-3D multi-view image evaluation")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Path to the LLaVA-3D model checkpoint")
    parser.add_argument("--model_base", type=str, default=None,
                        help="Base model path (if using LoRA)")
    parser.add_argument("--upd_text_folder_path", type=existing_dir, required=True, 
                        help="Path to the upd_text/ folder.")
    parser.add_argument("--upd_version_name", type=str, required=True, 
                        help="Name of the upd version (e.g., 'v1').")
    parser.add_argument("--upd_version_name_subfolder", type=str, required=True, 
                        help="Subfolder name for the upd version (e.g., 'standard').")
    parser.add_argument("--video_path", type=existing_dir, required=True, 
                        help="Path to the video data folder containing dirs identifier/scene/")
    parser.add_argument("--scene_list_txt_file_path", type=existing_file, required=True, 
                        help="Path to the text file containing scene identifiers and scene names.")
    parser.add_argument("--json_tag", type=str, required=False, 
                        help="Optional tag to include in the output JSON filename", 
                        default=None)
    args = parser.parse_args()

    # Check that the passed paths are valid
    updtext_versionfolder_subfolder_path = os.path.join(
        args.upd_text_folder_path,
        args.upd_version_name,
        args.upd_version_name_subfolder
    )
    if not os.path.isdir(updtext_versionfolder_subfolder_path):
        raise ValueError(f"Error: '{updtext_versionfolder_subfolder_path}' is not a valid folder path.")

    # Initialize model
    model, tokenizer, processor, keywords, mm_use_im_start_end, conv = init_model(args)

    # Run batch inference
    inference(
        scene_list_txt_file_path=args.scene_list_txt_file_path,
        updtext_versionfolder_subfolder_path=updtext_versionfolder_subfolder_path,
        video_path=args.video_path,
        upd_subset_name=args.upd_version_name_subfolder,
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        keywords=keywords,
        mm_use_im_start_end=mm_use_im_start_end,
        conv_template=conv,
        json_tag=args.json_tag
    )