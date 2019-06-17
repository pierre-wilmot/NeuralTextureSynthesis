# NeuralTextureSynthesis
Code for "[Stable and Controllable Neural Texture Synthesis and Style Transfer Using Histogram Losses](https://arxiv.org/pdf/1701.08893v1.pdf)"
## Notes:

- The histogram functionality relies on a custom cuda module, so you'll need a cuda GPU to run the code.
- Tested on GTX1080, top VRAM usage is just under 2700MB, runtime is 10 mins per image.
- Code differs from the paper as the loss balancing is done with "magic numbers" rather than using the gradient clipping described in the paper.
- Code doesn't implement the TV loss (total variation), as results didn't appear to have high frequency noise.

## Usage:

The code contains a custom C++/Cuda module that gets compiled on the fly, in order to do that you need to use the same compiller as the one used to compile your release of PyTorch. On my machine, that's g++.

To launch the program :
``` CXX=g++ python main.py src/*```

This will run all images in src folder with and without histogram and generate a results.html page.

## Sample results:

https://pierre-wilmot.github.io/NeuralTextureSynthesis/
