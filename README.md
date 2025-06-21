# ComfyUI GFPGAN

This is a custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that enables face restoration using [GFPGAN](https://github.com/TencentARC/GFPGAN).

## Installation

1.  **Navigate to the Custom Nodes Directory**  
    ```bash
    cd /path/to/your/ComfyUI/custom_nodes/
    ```

2.  **Clone the Repository**
    ```bash
    git clone https://github.com/lucak5s/comfyui_gfpgan.git
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r comfyui_gfpgan/requirements.txt -c constraints.txt
    ```
    > **Note:** The use of the `constraints.txt` file is necessary because `gfpgan` depends on the standard `basicsr` package, which is not compatible with the latest versions of PyTorch.


## Usage

Start ComfyUI and look for the node named **"GFPGAN Face Restore."**  
The required face detection and face restoration models will be downloaded automatically and placed in the following directories:

- `ComfyUI/models/face_detection`  
- `ComfyUI/models/face_restoration`
