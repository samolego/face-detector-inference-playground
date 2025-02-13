mod config;
mod detection;
mod drawing;

use clap::Parser;
use image::imageops::FilterType;
use ort::CPUExecutionProvider;
use std::{fs, path::PathBuf, rc::Rc};

#[derive(Parser, Debug, Clone)]
#[command(
    name = "face-detection",
    author = "samo_lego",
    version,
    about = "Face detection using buffalo model",
    long_about = "A face detection tool that processes images using AI. \
                  Can process single images or entire directories and supports various output options.",
    after_help = "Examples:\n\
                  Process using default model:\n\
                  face-detection -i image.jpg -o output.jpg -t 0.6\n\n\
                  Process using custom model:\n\
                  face-detection -i image.jpg -o output.jpg -m path/to/model.onnx\n\n\
                  Process directory with custom model:\n\
                  face-detection -i ./images -o ./processed -m path/to/model.onnx -v"
)]
struct Args {
    /// Input path (file or directory)
    #[arg(
        short,
        long,
        help = "Path to input image or directory containing images",
        long_help = "Specify the path to a single image file or a directory containing multiple images. \
                     Supported formats: jpg, jpeg, png, gif, bmp"
    )]
    input: String,

    /// Output path (file for single image, directory for multiple images)
    #[arg(
        short,
        long,
        help = "Path for output (file or directory). If not specified, no images will be saved",
        long_help = "Where to save the processed images. For single image input, specify a file path. \
                     For directory input, specify an output directory. \
                     If not specified, no images will be saved"
    )]
    output: Option<String>,

    /// Confidence threshold (0.0 to 1.0)
    #[arg(
        short,
        long,
        default_value_t = 0.5,
        help = "Detection confidence threshold",
        long_help = "Minimum confidence threshold for face detection (0.0 to 1.0). \
                     Higher values result in fewer but more confident detections"
    )]
    threshold: f32,

    /// Path to ONNX model file
    #[arg(
        short,
        long,
        help = "Path to ONNX model file",
        long_help = "Path to the ONNX model file for face detection. \
                         If not specified, uses the default model at './det_10g.onnx'"
    )]
    model: Option<String>,

    /// Print detection results to console
    #[arg(
        short,
        long,
        action = clap::ArgAction::SetTrue,
        help = "Print detection details to console",
        long_help = "Print detailed information about detected faces to console, \
                     including confidence scores and bounding box coordinates"
    )]
    verbose: bool,
}

fn process_single_image(
    input_path: &PathBuf,
    output_path: Option<&PathBuf>,
    model: &detection::model::DetectionModel,
    config: &Rc<config::Config>,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Processing: {}", input_path.display());

    let img = image::open(input_path)?;
    let resized = img.resize_exact(640, 640, FilterType::CatmullRom);
    let resized_rgb = resized.to_rgb8();

    let input_array = detection::preprocessing::preprocess_image(&resized_rgb, &config)?;
    let results = model.predict(input_array)?;

    // Process results
    let detections = detection::process_detections(&results, args.threshold)?;

    // Print detections if verbose mode is enabled
    if args.verbose {
        println!("Detections for {}:", input_path.display());
        for detection in &detections {
            println!(
                "  Confidence: {:.2}, Bbox: {:?}",
                detection.confidence, detection.bbox
            );
        }
    }

    // Save output image if path is provided
    if let Some(out_path) = output_path {
        let mut output_image = resized_rgb.clone();
        detection::draw_detections(&mut output_image, &detections)?;
        fs::create_dir_all(out_path.parent().unwrap_or(&PathBuf::from(".")))?;
        output_image.save(out_path)?;
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Initialize ONNX Runtime
    ort::init()
        .with_execution_providers([CPUExecutionProvider::default().build()])
        .commit()?;

    // Create config with updated threshold
    let mut config = config::Config::default();

    config.confidence_threshold = args.threshold;
    if let Some(model_path) = args.model.as_ref() {
        config.model_path = model_path.clone();
    }

    let config = Rc::new(config);

    let model = detection::model::DetectionModel::new(config.clone())?;

    let input_path = PathBuf::from(&args.input);

    if input_path.is_file() {
        // Process single file
        let output_path = args.output.as_ref().map(|p| PathBuf::from(p));
        process_single_image(&input_path, output_path.as_ref(), &model, &config, &args)?;
    } else if input_path.is_dir() {
        // Process directory
        let output_dir = args.output.as_ref().map(|p| PathBuf::from(p));

        // Create output directory if specified
        if let Some(dir) = &output_dir {
            fs::create_dir_all(dir)?;
        }

        for entry in fs::read_dir(input_path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && is_image_file(&path) {
                let output_path = output_dir.as_ref().map(|dir| {
                    let stem = path
                        .file_stem()
                        .unwrap_or_default()
                        .to_str()
                        .unwrap_or_default();
                    let ext = path
                        .extension()
                        .unwrap_or_default()
                        .to_str()
                        .unwrap_or_default();
                    dir.join(format!("{}_detections.{}", stem, ext))
                });

                process_single_image(&path, output_path.as_ref(), &model, &config, &args)?;
            }
        }
    } else {
        return Err("Input path does not exist".into());
    }

    Ok(())
}

fn is_image_file(path: &PathBuf) -> bool {
    let extensions = ["jpg", "jpeg", "png", "gif", "bmp"];
    if let Some(ext) = path.extension() {
        if let Some(ext_str) = ext.to_str() {
            return extensions.contains(&ext_str.to_lowercase().as_str());
        }
    }
    false
}
