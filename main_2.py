"""
Alternative entry point for the Cells Calculator application.

This module provides a simple interface for running trained models on images
without the GUI. It's designed for batch processing and scripting workflows.

Usage:
    from main_2 import run_model_on_image
    
    results = run_model_on_image(
        model_path="trainedmodels/YOLO11x-512-seg.pt",
        image_path="testimages/sample.jpg",
        model_type="yolo",
        confidence=0.3,
        iou=0.6
    )
"""

import json
from collections import OrderedDict
from pathlib import Path
import importlib
import sys
import traceback


def load_model(model_path: str, model_type: str, **kwargs):
    """
    Load a model from the specified path.
    
    Args:
        model_path (str): Path to the model file
        model_type (str): Type of model - 'yolo', 'cellpose', 'instanseg', 'stardist', 'cellcounter'
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Loaded model instance
        
    Raises:
        ValueError: If model_type is not supported
        FileNotFoundError: If model file doesn't exist
    """
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_type = model_type.lower()
    
    if model_type == "yolo":
        from ultralytics import YOLO
        return YOLO(model_path, task="segment")
    
    elif model_type == "cellpose":
        from model.CellposeSegmenter import CellposeSegmenter
        # Добавили model_data={}
        model = CellposeSegmenter(model_path, object_size={}, model_data={}) 
        return model
    
    elif model_type == "instanseg":
        from model.InstanSegSegmenter import InstansegSegmenter
        # Добавили model_data={}
        model = InstansegSegmenter(model_path, object_size={}, model_data={})
        return model
    
    elif model_type == "stardist":
        from model.StardistSegmenter import StardistSegmenter
        # Добавили model_data={}
        model = StardistSegmenter(model_path, object_size={}, model_data={})
        return model
    
    elif model_type == "cellcounter":
        from model.CellCounter import CellCounter
        # Добавили model_data={}
        model = CellCounter(model_path, object_size={}, model_data={})
        return model
    else:
        raise ValueError(f"Unsupported model type: {model_type}. "
                        f"Supported types: yolo, cellpose, instanseg, stardist, cellcounter")


def run_model_on_image(model_path: str, image_path: str, model_type: str = "yolo", **inference_kwargs):
    """
    Run a trained model on an image and return results.
    
    This is the main function for inference without the GUI.
    
    Args:
        model_path (str): Path to the trained model
        image_path (str): Path to the input image
        model_type (str): Type of model - 'yolo', 'cellpose', 'instanseg', 'stardist', 'cellcounter'
                         Default: 'yolo'
        **inference_kwargs: Additional arguments for inference (e.g., conf=0.3, iou=0.6)
        
    Returns:
        dict: Dictionary containing:
            - 'success' (bool): Whether inference was successful
            - 'results' (object): Raw model results
            - 'message' (str): Status message
            - 'model': The loaded model instance
            - 'error' (str): Error message if unsuccessful
            
    Example:
        >>> results = run_model_on_image(
        ...     model_path="trainedmodels/YOLO11x-512-seg.pt",
        ...     image_path="testimages/sample.jpg",
        ...     model_type="yolo",
        ...     conf=0.3,
        ...     iou=0.6
        ... )
        >>> if results['success']:
        ...     print(f"Found {len(results['results'])} objects")
        ... else:
        ...     print(f"Error: {results['error']}")
    """
    
    try:

        if not Path(image_path).exists():
            return {
                'success': False,
                'results': None,
                'message': f'Image file not found: {image_path}',
                'model': None,
                'error': f'Image file not found: {image_path}'
            }
        
        print(f"Loading model from: {model_path}")
        model = load_model(model_path, model_type)
        print(f"Model loaded successfully (type: {model_type})")
        
        print(f"Running inference on: {image_path}")
        
        if model_type == "yolo":

            results = model(image_path, **inference_kwargs)
            return {
                'success': True,
                'results': results,
                'message': f'Inference completed successfully on {image_path}',
                'model': model,
                'error': None
            }
        
        elif model_type in ["cellpose", "instanseg", "stardist", "cellcounter"]:
            
            if hasattr(model, 'predict'):
                results = model.predict(image_path, **inference_kwargs)
            elif hasattr(model, 'calculate'):
                results = model.calculate(image_path)
            elif hasattr(model, 'count_x20'):
                results = model.count_x20(input_image=image_path, tracking=True, plot=False)
            else:
                raise AttributeError(f"Класс {type(model).__name__} не имеет методов 'predict', 'calculate' или 'count_x20'")
                
            return {
                'success': True,
                'results': results,
                'message': f'Inference completed successfully on {image_path}',
                'model': model,
                'error': None
            }
        
    except Exception as e:
        error_msg = f"Error during inference: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {
            'success': False,
            'results': None,
            'message': 'Inference failed',
            'model': None,
            'error': error_msg
        }


def run_batch_inference(model_path: str, image_directory: str, model_type: str = "yolo", 
                       output_directory: str = None, **inference_kwargs):
    """
    Run inference on multiple images in a directory.
    
    Args:
        model_path (str): Path to the trained model
        image_directory (str): Directory containing images
        model_type (str): Type of model. Default: 'yolo'
        output_directory (str): Directory to save results. If None, creates 'inference_output'
        **inference_kwargs: Additional arguments for inference
        
    Returns:
        dict: Summary of batch processing results
        
    Example:
        >>> summary = run_batch_inference(
        ...     model_path="trainedmodels/YOLO11x-512-seg.pt",
        ...     image_directory="testimages/",
        ...     model_type="yolo",
        ...     output_directory="inference_results/",
        ...     conf=0.3
        ... )
    """
    
    img_dir = Path(image_directory)
    if not img_dir.exists():
        return {
            'success': False,
            'message': f'Image directory not found: {image_directory}',
            'processed': 0,
            'failed': 0
        }
    
    if output_directory is None:
        output_directory = "inference_output"
    
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in img_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    results_summary = {
        'success': True,
        'total_images': len(image_files),
        'processed': 0,
        'failed': 0,
        'results': [],
        'output_directory': str(out_dir)
    }
    
    print(f"Found {len(image_files)} images to process")
    
    try:
        model = load_model(model_path, model_type)
    except Exception as e:
        results_summary['success'] = False
        results_summary['error'] = str(e)
        return results_summary
    
    for idx, img_file in enumerate(image_files, 1):
        print(f"Processing {idx}/{len(image_files)}: {img_file.name}")
        
        result = run_model_on_image(
            model_path=model_path,
            image_path=str(img_file),
            model_type=model_type,
            **inference_kwargs
        )
        
        if result['success']:
            results_summary['processed'] += 1
            results_summary['results'].append({
                'image': img_file.name,
                'success': True
            })
        else:
            results_summary['failed'] += 1
            results_summary['results'].append({
                'image': img_file.name,
                'success': False,
                'error': result['error']
            })
    
    return results_summary


if __name__ == '__main__':
    """
    Example usage when running this file directly.
    
    Run with:
        python main_2.py <model_path> <image_path> [model_type] [confidence] [iou]
    
    Example:
        python main_2.py trainedmodels/YOLO11x-512-seg.pt testimages/sample.jpg yolo 0.3 0.6
    """
    
    if len(sys.argv) < 3:
        print("Usage: python main_2.py <model_path> <image_path> [model_type] [confidence] [iou]")
        print("\nExample:")
        print("  python main_2.py trainedmodels/YOLO11x-512-seg.pt testimages/sample.jpg yolo 0.3 0.6")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    model_type = sys.argv[3] if len(sys.argv) > 3 else "yolo"
    
    inference_kwargs = {}
    if len(sys.argv) > 4:
        try:
            inference_kwargs['conf'] = float(sys.argv[4])
        except ValueError:
            pass
    if len(sys.argv) > 5:
        try:
            inference_kwargs['iou'] = float(sys.argv[5])
        except ValueError:
            pass
    
    print(f"Running inference...")
    print(f"  Model: {model_path}")
    print(f"  Image: {image_path}")
    print(f"  Model Type: {model_type}")
    if inference_kwargs:
        print(f"  Inference kwargs: {inference_kwargs}")
    print()
    
    results = run_model_on_image(model_path, image_path, model_type, **inference_kwargs)
    
    print(f"\nResults:")
    print(f"  Success: {results['success']}")
    print(f"  Message: {results['message']}")
    
    if not results['success']:
        print(f"  Error: {results['error']}")
        sys.exit(1)
    else:
        print(f"  Raw results: {results['results']}")
