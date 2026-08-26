# openpilot.distill

You too, can train state-of-the-art autonomous driving models!

With a comma four, and a few logged segments, fine-tune the openpilot driving model to your own driving style.

Drop your segments video files and rlogs into the `data/` folder.

## Step 1: Run a simple localization pipeline.

Run the localizer.

## Step 2: Fine tune the driving model using supervised learning.

Fine tune the driving model using localized plan and image auto-encoder losses.

## Step 3: Fine tune the driving model in the simulator.

DAgger style fine tuning using a learned world model.

We don't have big tech processing power, so we need to be creative at every step.
