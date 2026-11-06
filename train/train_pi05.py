"""PI0.5 LoRA baseline: fine-tune by default, evaluate with --evaluate.

Uses the same entry/config/algo/helpers layout as the GR00T baseline.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.train_configs.pi05_config import parse_args


def main():
    args = parse_args()
    if args.evaluate:
        from utils.pi05_utils.eval import evaluate

        evaluate(args)
    else:
        from utils.pi05_utils.training import finetune

        finetune(args)


if __name__ == "__main__":
    main()
