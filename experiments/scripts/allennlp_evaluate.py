import json
import shutil
import sys

from allennlp.commands import main


def allennllp_evaluate(model_gz_path: str, dataset: str, output_file_path: str, cuda_device: int = 0):
    # See https://guide.allennlp.org/debugging#3

    # Assemble the command into sys.argv
    sys.argv = [
        "allennlp",  # command name, not used by main
        "evaluate",
        model_gz_path,
        dataset,
        "--file-friendly-logging",
        "--cuda-device", cuda_device,
        "--output-file", output_file_path,
        "--include-package", "experiments.allennlp_extensions",
    ]

    main()
