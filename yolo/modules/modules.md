# Modules

## Content
- [Squeeze And Excitation](Squeeze-And-Excitation)

## Squeeze And Excitation
#### Image Model Blocks
<div align="center">
    <img src="yolo.ai/images/modules/squeeze_and_excitation.png">
</div>

- The block has a convolutional block as an input.
- Each channel is "squeezed" into a single numeric value using average pooling.
- A dense layer followed by a ReLU adds non-linearity and output channel complexity is reduced by a ratio.
- Another dense layer followed by a sigmoid gives each channel a smooth gating function.
- Finally, we weight each feature map of the convolutional block based on the side network; the "excitation".
