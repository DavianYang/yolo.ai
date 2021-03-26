# Modules

## Content
- [Squeeze And Excitation](#Squeeze-And-Excitation)
- [Inverted Residuals](#Inverted-Residuals)

## Squeeze And Excitation

<div align="center">
    <img src="../images/modules/squeeze_and_excitation.png">
</div>

- Given convolutional block as an input.
- Each channel is "squeezed" into a single numeric value using average pooling.
```H x W x C-> 1 x 1 x C```
- Two convolutional layer followed by SiLU which give smooth and non-monotonic attribute. Which give as
```1 x 1 x C -> [0, 1]```
- This way, the network knows which channels should be prioritized. Eg. 0.1 give 10% of.
> Swish/SiLU outperforms ReLU by a large margin in the range between 40 and 50 layers when optimization becomes difficult.
- Check details of SiLU for [here](https://medium.com/@neuralnets/swish-activation-function-by-google-53e1ea86f820)


## Inverted Residuals

<div align="center">
    <img src="../images/modules/inverted_residual.png">
</div>

- In original residual block, it follow ```wide -> narrow -> wide``` approach. Eg. ```3x3 -> 1x1 -> 3x3```
- In inverted residual block, it follow ```narrow -> wide -> narrow``` approach. Eg. ```1x1 -> 3x3 -> 1x1```
> In Efficient Network, SEBlocks are in 1x1 -> 3x3 -> here -> 1x1
- Check details of Inverted Residuals for [here](https://medium.com/@luis_gonzales/a-look-at-mobilenetv2-inverted-residuals-and-linear-bottlenecks-d49f85c12423)