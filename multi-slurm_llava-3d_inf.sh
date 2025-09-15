subfolders=(
    "standard"
    "open_ended"
    "open_ended_additional_instruction"
    "aad_base"
    "aad_additional_option"
    "aad_additional_instruction"
    "iasd_base"
    "iasd_additional_option"
    "iasd_additional_instruction"
    "ivqd_base"
    "ivqd_additional_option"
    "ivqd_additional_instruction"
)

# Use local checkpoint - adjust path as needed
MODEL_DIR="$(realpath ./checkpoints/LLaVA-3D-7B)"
if [ ! -d "$MODEL_DIR" ]; then
    echo "[multi-slurm] ERROR: Model directory $MODEL_DIR not found" >&2
    echo "[multi-slurm] Available checkpoints:" >&2
    ls -la ./checkpoints/ 2>/dev/null || echo "No checkpoints/ directory found" >&2
    exit 1
fi

echo "[multi-slurm] Using model: $MODEL_DIR" >&2

for subfolder in "${subfolders[@]}"; do
    echo "[multi-slurm] Submitting job for subfolder: $subfolder" >&2
    sbatch slurm_llava-3d_inf.sh \
        --model_name "$MODEL_DIR" \
        --upd_text_folder_path /project/3dllms/melgin/UPD-3D/upd_text/ \
        --video_path /gscratch/melgin/3d-grand_unzipped/3D-FRONT/ \
        --scene_list_txt_file_path /project/3dllms/melgin/UPD-3D/pcl_lists/3D-FRONT_test.txt \
        --json_tag llava3d \
        --upd_version_name_subfolder "$subfolder" \
        --upd_version_name "3D-FRONT" 
done