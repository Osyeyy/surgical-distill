from datasets import load_dataset

ds = load_dataset("ClimbMix/climbmix", split="train")  # verify exact name on HF

with open("data/calibration.txt", "w") as f:
    for i, row in enumerate(ds):
        if i >= 500:
            break
        # check what the text field is called
        text = row.get("text") or row.get("prompt") or row.get("instruction") or ""
        if text.strip():
            f.write(text.strip()[:512] + "\n")  # cap length so forward passes are fast