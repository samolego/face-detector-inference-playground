
# Face Detection using Buffalo model in Rust

**Warning: this is just a toy project for playing around with buffalo face recognition model.**

A ~~blazingly-fast~~  face detection tool built in Rust using ONNX Runtime.
This tool can process both single images and directories of images, providing flexible output options for detection results.

### Building from source

```bash
git clone https://github.com/your-username/face-detection-rs.git
cd face-detection-rs
cargo build --release
```

## Usage

Obtain [buffalo model from InsightFace](https://github.com/deepinsight/insightface/releases).
Note: this was tested with buffalo_l.
Extract the det_10g.onnx file from the downloaded zip file and place it in the same directory as the executable.

### Basic Command Structure

```bash
face-detection -i <INPUT> [-o <OUTPUT>] [-t <THRESHOLD>] [-v]
```

### Command Line Options

- `-i, --input <PATH>`: Input image or directory path (required)
- `-o, --output <PATH>`: Output file or directory path (optional)
- `-t, --threshold <FLOAT>`: Detection confidence threshold (default: 0.5)
- `-v, --verbose`: Print detection details to console
- `-h, --help`: Show help information
- `-V, --version`: Show version information

### Examples

1. Process a single image and save with detection boxes:
```bash
face-detection -i image.jpg -o output.jpg -t 0.6
```

2. Process an image and only print detections:
```bash
face-detection -i image.jpg -v
```

3. Process all images in a directory and save results:
```bash
face-detection -i ./images -o ./processed -v
```

4. Process directory and only print detections:
```bash
face-detection -i ./images -v
