# FastVLM

Demonstrates the performance of **FastVLM** models for on-device, visual question answering. 

<table>
<tr>
    <td><img src="../docs/fastvlm-counting.gif" alt="FastVLM - Counting"></td>
    <td><img src="../docs/fastvlm-handwriting.gif" alt="FastVLM - Handwriting"></td>
    <td><img src="../docs/fastvlm-emoji.gif" alt="FastVLM - Emoji"></td>
</tr>
</table>

## Features

- FastVLM runs on iOS (18.2+) and macOS (15.2+).
- View Time-To-First-Token (TTFT) with every inference.
- All predictions are processed privately and securely using on-device models.

### Flexible Prompting

<img src="../docs/fastvlm-flexible_prompts.png" alt="Flexible prompting" style="width:66%;">

The app includes a set of built-in prompts to help you get started quickly. Tap the **Prompts** button in the top-right corner to explore them. Selecting a prompt will immediately update the active input. To create new prompts or edit existing ones, choose **Customize…** from the **Prompts** menu.

## Pretrained Model Options

There are 3 pretrained sizes of FastVLM to choose from:

- **FastVLM 0.5B**: Small and fast - great for mobile devices where speed matters.
- **FastVLM 1.5B**: Well balanced - great for larger devices where speed and accuracy matters.
- **FastVLM 7B**: Fast and accurate - ideal for situations where accuracy matters over speed.

To download any FastVLM listed above, use one of the provided scripts. The scripts download the model from the web and place it in the appropriate location.

- [`get_pretrained_mlx_model.sh`](get_pretrained_mlx_model.sh): Uses `wget` to download the model.
- [`get_pretrained_mlx_model_curl.sh`](get_pretrained_mlx_model_curl.sh): Uses `curl` to download the model. Use this if you do not have `wget` installed.

Once a model has been downloaded using the steps below, no additional steps are needed to build the app in Xcode.

To explore how the other models work for your use-case, simply re-run the appropriate script with the new model selected, follow the prompts, and rebuild your app in Xcode.

### Download Instructions

The download script comes in two versions: one using `wget` and another using `curl`. Choose the one appropriate for your system.

1. Make the chosen script executable

```shell
# If you have wget
chmod +x app/get_pretrained_mlx_model.sh

# If you have curl
chmod +x app/get_pretrained_mlx_model_curl.sh
```

2. Download FastVLM

```shell
# Using wget
app/get_pretrained_mlx_model.sh --model 0.5b --dest app/FastVLM/model

# Using curl
app/get_pretrained_mlx_model_curl.sh --model 0.5b --dest app/FastVLM/model
```

3. Open the app in Xcode, Build, and Run.

### Custom Model

In addition to pretrained sizes of FastVLM, you can further quantize or fine-tune FastVLM to best fit their needs. To learn more, check out our documentation on how to [`export the model`](../model_export#export-vlm).
Please clear existing model in `app/FastVLM/model` before downloading or copying a new model. 
