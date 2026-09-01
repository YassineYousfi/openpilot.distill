# openpilot.distill

You too, can train state-of-the-art autonomous driving models!

With a comma four, and a few logged segments, distill the openpilot driving model to a new model and make it drive your own driving style.

Drop your segments video files and rlogs into the `data/` folder.

You don't have a comma four, no problem! You can still download videos and targets from https://huggingface.co/datasets/commaai/comma1M
and train your own driving models.

## Step 1: Run a simple localization pipeline

Run the localizer.

<img width="512" height="256" alt="2333e255f1de1fe83ea5975b24a3cc67_frame_700" src="https://github.com/user-attachments/assets/395683ff-7d96-4e07-bf6e-12a3969b487f" />

## Step 2: Fine tune the driving model using supervised learning

Train a smaller driving model using the localized targets, and distillation targets from the openpilot model.

https://api.wandb.ai/links/yassiney/qme6ln4z

## Step 3: Fine tune the driving model in the simulator

DAgger style post-training in a learned world model as a simulator.


Have fun!