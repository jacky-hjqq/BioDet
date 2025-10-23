import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Remove 'segmentation' key from detection results JSON.")
    parser.add_argument("--dataset_name", required=True, help="Dataset name (e.g., ipd)")
    parser.add_argument("--segmentation_model", required=True, help="Segmentation model name (e.g., sam)")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    segmentation_model = args.segmentation_model

    detection_res_path = Path(f"log/ISM_{segmentation_model}/result_{dataset_name}.json")
    bop_detection_res_path = Path(f"log/ISM_{segmentation_model}/bop_result_{dataset_name}.json")

    if not detection_res_path.exists():
        print(f"File not found: {detection_res_path}")
        return

    print(f"Loading: {detection_res_path}")
    with open(detection_res_path, "r") as f:
        detection_res = json.load(f)

    for item in detection_res:
        if "segmentation" in item:
            del item["segmentation"]

    print(f"Saving cleaned results to: {bop_detection_res_path}")
    bop_detection_res_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bop_detection_res_path, "w") as f:
        json.dump(detection_res, f, indent=4)

if __name__ == "__main__":
    main()

