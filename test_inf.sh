# python llava/eval/run_llava_3d.py \
#     --model-path ChaimZhu/LLaVA-3D-7B \
#     --image-file https://llava-vl.github.io/static/images/view.jpg \
#     --query "What are the things I should be cautious about when I visit here?"

# python llava/eval/run_llava_3d.py \
#     --model-path ChaimZhu/LLaVA-3D-7B \
#     --image-file https://llava-vl.github.io/static/images/view.jpg \
#     --query "What is this?"

sbatch slurm_llava-3d_inf.sh \
    --model_name ./checkpoints/LLaVA-3D-7B \
    --upd_text_folder_path /project/3dllms/melgin/UPD-3D/upd_text/ \
    --upd_version_name "3D-FRONT" \
    --upd_version_name_subfolder standard \
    --video_path /project/3dllms/melgin/datasets/3d-grand_unzipped_gpt-5-nano/3D-FRONT/ \
    --scene_list_txt_file_path /project/3dllms/melgin/UPD-3D/pcl_lists/3D-FRONT_test.txt \
    --json_tag llava3d_test