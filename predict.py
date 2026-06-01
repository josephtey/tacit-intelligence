import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="Predict pipette well position from video clips"
    )
    parser.add_argument("fpv_video", help="Path to FPV video file")
    parser.add_argument("topview_video", help="Path to Topview video file")
    args = parser.parse_args()

    from src.e2e_pipeline import EndToEndPipeline

    pipeline = EndToEndPipeline()
    result = pipeline.predict(args.fpv_video, args.topview_video)

    # Strip internal stage details for clean output
    output = {
        "clip_id_FPV": result["clip_id_FPV"],
        "clip_id_Topview": result["clip_id_Topview"],
        "wells_prediction": result["wells_prediction"],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
