MODEL_NAME_S="resunet_combined_scaled_1000"

# Each dataset is defined as: "<image_dir>::<avg_radius>"
DATASETS=(
    "data/raw/1_c_elegans_nuclei/test/images::6.0"
    "data/raw/2_livernuclei/test/images::12.0"
    "data/raw/3_mesoSPIM_dataset/test/images::3.5"
    "data/raw/4_Mouse_NucMM-M/test/images::6.5"
    "data/raw/5_Zebrafish_NucMM-Z/test/images::7.0"
    "data/raw/6_Drosophila_denoised/test/images::13.5"
    "data/raw/7_liver_hcc_dataset/test/images::14.0"   
)

echo "-----------------------------------"
echo " Running nuclei predictions"
echo " Model: $MODEL_NAME_S"
echo "-----------------------------------"

# Iterate over all datasets
for entry in "${DATASETS[@]}"; do
    IMAGE_DIR="${entry%%::*}"   # everything before '::'
    AVG_RADIUS="${entry##*::}"  # everything after '::'

    echo "-----------------------------------"
    echo "Running nuclei prediction"
    echo "Dataset: $IMAGE_DIR"
    echo "avg_radius: $AVG_RADIUS"
    echo "-----------------------------------"

    # Run nuclei prediction
    python -m nuclei_prediction "$MODEL_NAME_S" "$IMAGE_DIR" --avg_radius "$AVG_RADIUS"

    if [ $? -eq 0 ]; then
        echo "Completed prediction for: $IMAGE_DIR"
    else
        echo "Prediction failed for: $IMAGE_DIR"
    fi

    echo "-----------------------------------"
done

echo "All predictions completed from scaled model!"
