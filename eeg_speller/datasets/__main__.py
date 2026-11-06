from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_speller.datasets.bci3 import import_bci3
from eeg_speller.datasets.samara import import_samara
from eeg_speller.evaluation.flash import evaluate
from eeg_speller.physiology.session_p300 import P300FlashDecoder


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m eeg_speller.datasets")
    sub = parser.add_subparsers(dest="command", required=True)

    imp = sub.add_parser("import")
    imp.add_argument("--dataset", required=True, choices=("bci3", "samara"))
    imp.add_argument("--root", required=True, help="mat file or directory of mats")
    imp.add_argument("--subject", default=None)
    imp.add_argument("--split", default=None, choices=("train", "test"))
    imp.add_argument("--output", required=True)
    imp.add_argument("--eloc", default=None)
    imp.add_argument("--file-pattern", default="S*-P300_classic.mat")
    imp.add_argument("--split-mode", default="stratified", choices=("stratified", "chronological"))
    imp.add_argument("--seed", type=int, default=0)

    train = sub.add_parser("train")
    train.add_argument("--session", action="append", required=True)
    train.add_argument("--decoder", default="p300", choices=("p300",))
    train.add_argument("--output", required=True)
    train.add_argument("--channels", default=None, help="comma-separated channel names")
    train.add_argument("--baseline", nargs=2, type=float, default=None, metavar=("START", "STOP"))
    train.add_argument("--response", nargs=2, type=float, default=None, metavar=("START", "STOP"))
    train.add_argument("--band", nargs=2, type=float, default=(0.5, 15.0), metavar=("LOW", "HIGH"))

    ev = sub.add_parser("eval")
    ev.add_argument("--session", required=True)
    ev.add_argument("--decoder", default="p300", choices=("p300",))
    ev.add_argument("--model", required=True)

    args = parser.parse_args(argv)
    if args.command == "import":
        if args.dataset == "bci3":
            path = _bci_path(Path(args.root), args.subject, args.split)
            written = import_bci3(path, args.output, subject=args.subject, eloc=args.eloc)
            print(written)
        else:
            if not args.subject:
                raise SystemExit("samara import requires --subject")
            written = import_samara(args.root, args.output, subject=args.subject, pattern=args.file_pattern,
                                    mode=args.split_mode, base_seed=args.seed)
            for path in written:
                print(path)
        return
    if args.command == "train":
        channels = None if not args.channels else [item.strip() for item in args.channels.split(",") if item.strip()]
        decoder = P300FlashDecoder(args.output, channels=channels, baseline_s=args.baseline,
                                   response_s=args.response, bandpass_hz=tuple(args.band))
        model = decoder.fit([Path(path) for path in args.session])
        print(json.dumps({"model": str(Path(args.output)), "dataset": model["dataset_name"],
                          "train_subjects": model["train_subjects"]}, ensure_ascii=False))
        return
    record = evaluate(args.session, args.model)
    print(json.dumps(record, ensure_ascii=False, indent=2))


def _bci_path(root: Path, subject: str | None, split: str | None) -> Path:
    if root.is_file():
        return root
    if not subject or not split:
        raise SystemExit("bci3 import from a directory requires --subject and --split")
    name = f"Subject_{subject}_{'Train' if split == 'train' else 'Test'}.mat"
    path = root / name
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    return path


if __name__ == "__main__":
    main()
