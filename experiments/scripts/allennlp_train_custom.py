import json
import shutil
import sys

from allennlp.commands import main

def allennlp_train():
    # See https://guide.allennlp.org/debugging#3

    config_file = "experiments/train_configs/mt5.json"

    # Use overrides to train on CPU.
    overrides = json.dumps({"trainer": {"cuda_device": 0}})

    serialization_dir = "temp"

    # Training will fail if the serialization directory already
    # has stuff in it. If you are running the same training loop
    # over and over again for debugging purposes, it will.
    # Hence, we wipe it out in advance.
    # BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
    shutil.rmtree(serialization_dir, ignore_errors=True)

    # Assemble the command into sys.argv
    sys.argv = [
        "allennlp",  # command name, not used by main
        "train",
        config_file,
    # "--recover",
        "-s", serialization_dir,
        "--include-package", "experiments.allennlp_extensions",
        "-o", overrides,
    ]

    main()

allennlp_train()