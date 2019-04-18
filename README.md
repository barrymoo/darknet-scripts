Warning: Research Grade Scripts!
---

This repository contains some scripts for training custom models with [Darknet](https://github.com/AlexeyAB/darknet).
The scripts are __specific__ to our dataset and you are welcome to contact me with any questions.

### Getting set up

1. Generate images for your training (`./generate_images.py`)
    - Comment out the `image.resize()` to determine your image size
    - Adjust them down such that each is divisible by 32
    - Modify the resize line to these dimensions (you will need these dimensions later)
2. ...

### Running Darknet

- Training: `darknet detector train towhee.data cfg/towhee.cfg | tee training.log`
  - Generate loss: `./generate_loss.sh`, plot in gnuplot via `plot check.loss`
- Validation: `darknet detector map towhee.data cfg/towhee.cfg backup/towhee_last.weights`
