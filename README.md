# FastVLM: Efficient Vision Encoding for Vision Language Models

This is the official repository of
**[FastVLM: Efficient Vision Encoding for Vision Language Models](https://www.arxiv.org/abs/2412.13303). (CVPR 2025)**

[//]: # (![FastViTHD Performance]&#40;docs/acc_vs_latency_qwen-2.png&#41;)
<p align="center">
<img src="docs/acc_vs_latency_qwen-2.png" alt="Accuracy vs latency figure." width="400"/>
</p>

## Building with Current Packages (June 2026)

> **Note for developers:** This repository was published as a single frozen commit in May 2025.
> The Swift app no longer builds against current package versions due to breaking changes in
> `mlx-swift-lm 3.x` and the migration of `MLXLMCommon`/`MLXVLM` out of `mlx-swift-examples`
> into a new repository.
>
> A working fix is available on the
> [`fix/mlx-swift-lm-3x`](https://github.com/tmorales2000/ml-fastvlm/tree/fix/mlx-swift-lm-3x)
> branch of this fork, tested on macOS with M1 Pro and M4 Pro against:
> - `mlx-swift` 0.25.6
> - `mlx-swift-lm` 3.31.3
> - `swift-transformers` 1.3.3
> - `swift-huggingface` 0.9.0
>
> See the open [pull request](https://github.com/apple/ml-fastvlm/pull/78) and
> [issue](https://github.com/apple/ml-fastvlm/issues/77) on Apple's repo for details.
>
> **macOS note:** `get_models.sh` and `app/get_pretrained_mlx_model.sh` require `wget`,
> which is not installed by default on macOS. This fork patches both scripts to use `curl` instead.

### Highlights
* We introduce FastViTHD, a novel hybrid vision encoder designed to output fewer tokens and significantly reduce encoding time for high-resolution images.  
* Our smallest variant outperforms LLaVA-OneVision-0.5B with 85x faster Time-to-First-Token (TTFT) and 3.4x smaller vision encoder.
* Our larger variants using Qwen2-7B LLM outperform recent works like Cambrian-1-8B while using a single image encoder with a 7.9x faster TTFT.
* Demo iOS app to demonstrate the performance of our model on a mobile device.

<table>
<tr>
    <td><img src="docs/fastvlm-counting.gif" alt="FastVLM - Counting"></td>
    <td><img src="docs/fastvlm-handwriting.gif" alt="FastVLM - Handwriting"></td>
    <td><img src="docs/fastvlm-emoji.gif" alt="FastVLM - Emoji"></td>
</tr>
</table>

## Getting Started
We use LLaVA codebase to train FastVLM variants. In order to train or finetune your own variants, 
please follow instructions provided in [LLaVA](https://github.com/haotian-liu/LLaVA) codebase. 
We provide instructions for running inference with our models.   

### Setup
```bash
conda create -n fastvlm python=3.10
conda activate fastvlm
pip install -e .
```

### Model Zoo
For detailed information on various evaluations, please refer to our [paper](https://www.arxiv.org/abs/2412.13303).

| Model        | Stage |                                            Pytorch Checkpoint (url)                                             |
|:-------------|:-----:|:---------------------------------------------------------------------------------------------------------------:|
| FastVLM-0.5B |   2   | [fastvlm_0.5b_stage2](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage2.zip) |
|              |   3   | [fastvlm_0.5b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3.zip) |
| FastVLM-1.5B |   2   | [fastvlm_1.5b_stage2](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage2.zip) |
|              |   3   | [fastvlm_1.5b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage3.zip)  |
| FastVLM-7B   |   2   | [fastvlm_7b_stage2](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_7b_stage2.zip)  |
|              |   3   | [fastvlm_7b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_7b_stage3.zip)  |

To download all the pretrained checkpoints run the command below (note that this might take some time depending on your connection so might be good to grab ☕️ while you wait).

```bash
bash get_models.sh   # Files will be downloaded to `checkpoints` directory.
```

### Usage Example
To run inference of PyTorch checkpoint, follow the instruction below
```bash
python predict.py --model-path /path/to/checkpoint-dir \
                  --image-file /path/to/image.png \
                  --prompt "Describe the image."
```

### Inference on Apple Silicon
To run inference on Apple Silicon, pytorch checkpoints have to be exported to format 
suitable for running on Apple Silicon, detailed instructions and code can be found [`model_export`](model_export/) subfolder.
Please see the README there for more details.

For convenience, we provide 3 models that are in Apple Silicon compatible format: [fastvlm_0.5b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3_llm.fp16.zip), 
[fastvlm_1.5b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage3_llm.int8.zip), 
[fastvlm_7b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_7b_stage3_llm.int4.zip). 
We encourage developers to export the model of their choice with the appropriate quantization levels following 
the instructions in [`model_export`](model_export/).

### Inference on Apple Devices
To run inference on Apple devices like iPhone, iPad or Mac, see [`app`](app/) subfolder for more details.

## Citation
If you found this code useful, please cite the following paper:
```
@InProceedings{fastvlm2025,
  author = {Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokul Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel, Hadi Pouransari},
  title = {FastVLM: Efficient Vision Encoding for Vision Language Models},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2025},
}
```

## Acknowledgements
Our codebase is built using multiple opensource contributions, please see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for more details. 

## License
Please check out the repository [LICENSE](LICENSE) before using the provided code and
[LICENSE_MODEL](LICENSE_MODEL) for the released models.
