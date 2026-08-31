# openpilot.distill

You too, can train state-of-the-art autonomous driving models!

With a comma four, and a few logged segments, fine-tune the openpilot driving model to your own driving style.

Drop your segments video files and rlogs into the `data/` folder.

You don't have a comma four, no problem! You can still download videos and targets from https://huggingface.co/datasets/commaai/comma1M
and train your own driving models.

## Step 1: Run a simple localization pipeline

Run the localizer.

<img width="512" height="256" alt="2333e255f1de1fe83ea5975b24a3cc67_frame_700" src="https://github.com/user-attachments/assets/395683ff-7d96-4e07-bf6e-12a3969b487f" />

## Step 2: Fine tune the driving model using supervised learning

Fine tune the driving model using localized plan and image auto-encoder losses.

https://api.wandb.ai/links/yassiney/qme6ln4z

## Step 3: Fine tune the driving model in the simulator

DAgger style fine tuning using a learned world model.

We don't have big tech processing power, so we need to be creative at every step.
