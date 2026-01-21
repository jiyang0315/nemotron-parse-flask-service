import numpy as np
from PIL import Image
from typing import List, Optional, Union, Dict, Any
import torch
from torchvision import transforms as T
import albumentations as A
import cv2
import json

from transformers import ProcessorMixin, BaseImageProcessor, ImageProcessingMixin
from transformers.tokenization_utils_base import BatchEncoding
from transformers.image_utils import ChannelDimension, ImageInput, PILImageResampling, infer_channel_dimension_format
from transformers.utils import TensorType


class NemotronParseImageProcessor(BaseImageProcessor, ImageProcessingMixin):
    """
    Image processor for NemotronParse model.
    
    This processor inherits from BaseImageProcessor to be compatible with transformers AutoImageProcessor.
    """
    
    model_input_names = ["pixel_values"]
    
    def __init__(
        self,
        final_size: tuple = (2048, 1648),
        **kwargs,
    ):
        clean_kwargs = {}
        for k, v in kwargs.items():
            if not k.startswith('_') and k not in ['transform', 'torch_transform']:
                clean_kwargs[k] = v
        
        if 'size' in clean_kwargs:
            size_config = clean_kwargs.pop('size')
            if isinstance(size_config, dict):
                if 'longest_edge' in size_config:
                    longest_edge = size_config['longest_edge']
                    if isinstance(longest_edge, (list, tuple)):
                        final_size = tuple(int(x) for x in longest_edge)
                    else:
                        final_size = (int(longest_edge), int(longest_edge))
                elif 'height' in size_config and 'width' in size_config:
                    final_size = (int(size_config['height']), int(size_config['width']))
        
        super().__init__(**clean_kwargs)
        
        if isinstance(final_size, (list, tuple)) and len(final_size) >= 2:
            self.final_size = (int(final_size[0]), int(final_size[1]))
        elif isinstance(final_size, (int, float)):
            self.final_size = (int(final_size), int(final_size))
        else:
            self.final_size = (2048, 1648)  # Default fallback
        
        self._create_transforms()
    
    def _create_transforms(self):
        """Create transform objects (not serialized to JSON)."""
        if isinstance(self.final_size, (list, tuple)):
            self.target_height, self.target_width = int(self.final_size[0]), int(self.final_size[1])
        else:
            self.target_height = self.target_width = int(self.final_size)
        
        self.transform = A.Compose([
            A.PadIfNeeded(
                min_height=self.target_height, 
                min_width=self.target_width, 
                border_mode=cv2.BORDER_CONSTANT, 
                value=[255, 255, 255],
                p=1.0
            ),
        ])
        
        self.torch_transform = T.Compose([
            T.ToTensor(),
            # Note: Normalization is done within RADIO model
        ])

    def to_dict(self):
        """Override to exclude non-serializable transforms."""
        output = super().to_dict()
        output.pop('transform', None)
        output.pop('torch_transform', None)
        return output
    
    @classmethod
    def from_dict(cls, config_dict: dict, **kwargs):
        """Override to recreate transforms after loading."""
        config_dict = config_dict.copy()
        config_dict.pop('transform', None)
        config_dict.pop('torch_transform', None)
        
        # Clean any problematic entries
        for key in list(config_dict.keys()):
            if key.startswith('_') or config_dict[key] is None:
                config_dict.pop(key, None)
        
        # Ensure numeric types are correct
        if 'final_size' in config_dict:
            final_size = config_dict['final_size']
            if isinstance(final_size, (list, tuple)):
                config_dict['final_size'] = tuple(int(x) for x in final_size)
        
        try:
            return cls(**config_dict, **kwargs)
        except Exception as e:
            print(f"Warning: Error in from_dict: {e}")
            print("Using default parameters...")
            return cls(**kwargs)
    
    def save_pretrained(self, save_directory, **kwargs):
        """Save image processor configuration."""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save preprocessor config in standard HuggingFace format
        config = {
            "feature_extractor_type": "NemotronParseImageProcessor",
            "image_processor_type": "NemotronParseImageProcessor", 
            "processor_class": "NemotronParseImageProcessor",
            "size": {
                "height": self.final_size[0],
                "width": self.final_size[1],
                "longest_edge": self.final_size
            },
            "final_size": self.final_size,
        }
        
        config_path = os.path.join(save_directory, "preprocessor_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def _resize_with_aspect_ratio(self, image: np.ndarray) -> np.ndarray:
        """Resize image maintaining aspect ratio (exact replica of original LongestMaxSizeHW)."""
        height, width = image.shape[:2]
        max_size_height = self.target_height
        max_size_width = self.target_width
        
        # Original LongestMaxSizeHW algorithm from custom_augmentations.py
        aspect_ratio = width / height
        new_height = height
        new_width = width

        if height > max_size_height:
            new_height = max_size_height
            new_width = int(new_height * aspect_ratio)

        if new_width > max_size_width:
            new_width = max_size_width
            new_height = int(new_width / aspect_ratio)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    def _pad_to_size(self, image: np.ndarray) -> np.ndarray:
        """Pad image to target size with white padding (matches A.PadIfNeeded behavior)."""
        h, w = image.shape[:2]
        min_height, min_width = self.target_height, self.target_width
        
        pad_h = max(0, min_height - h)
        pad_w = max(0, min_width - w)
        
        if pad_h == 0 and pad_w == 0:
            return image
        
        if len(image.shape) == 3:
            padded = np.pad(
                image, 
                ((0, pad_h), (0, pad_w), (0, 0)), 
                mode='constant', 
                constant_values=255
            )
        else:
            padded = np.pad(
                image, 
                ((0, pad_h), (0, pad_w)), 
                mode='constant', 
                constant_values=255
            )
        
        return padded
    
    def preprocess(
        self,
        images: ImageInput,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess an image or batch of images for the NemotronParse model.
        
        Args:
            images: Input image(s)
            return_tensors: Type of tensors to return
        """
        
        # Ensure images is a list
        if not isinstance(images, list):
            images = [images]

        # Ensure images are RGB
        for i in range(len(images)):
            images[i] = images[i].convert('RGB')
        
        # Convert PIL images to numpy arrays if needed
        processed_images = []
        for image in images:
            if isinstance(image, Image.Image):
                image = np.asarray(image)
            processed_images.append(image)
        
        # Apply NemotronParse-specific transforms
        pixel_values = []
        for image in processed_images:
            processed_image = self._resize_with_aspect_ratio(image)
            
            if self.transform is not None:
                transformed = self.transform(image=processed_image)
                processed_image = transformed["image"]
            else:
                # Fallback: just pad to target size
                processed_image = self._pad_to_size(processed_image)
            
            pixel_values_tensor = self.torch_transform(processed_image)
            
            if pixel_values_tensor.shape[0] == 1:
                pixel_values_tensor = pixel_values_tensor.expand(3, -1, -1)
                
            pixel_values.append(pixel_values_tensor)
                        
        pixel_values = torch.stack(pixel_values)
        
        data = {"pixel_values": pixel_values}
        
        if return_tensors is not None:
            data = self._convert_output_format(data, return_tensors)
            
        return data
    
    def _convert_output_format(self, data: Dict[str, torch.Tensor], return_tensors: Union[str, TensorType]) -> Dict:
        """Convert output format based on return_tensors parameter."""
        if return_tensors == "pt" or return_tensors == TensorType.PYTORCH:
            return data
        elif return_tensors == "np" or return_tensors == TensorType.NUMPY:
            return {k: v.numpy() for k, v in data.items()}
        else:
            return data
    
    def __call__(self, images: Union[Image.Image, List[Image.Image]], **kwargs) -> Dict[str, torch.Tensor]:
        """Process images for the model (backward compatibility)."""
        return self.preprocess(images, **kwargs)


class NemotronParseProcessor(ProcessorMixin):
    
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("PreTrainedTokenizer", "PreTrainedTokenizerFast")
    
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        if image_processor is None:
            image_processor = NemotronParseImageProcessor(**kwargs)
               
        super().__init__(image_processor, tokenizer)
            
    
    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]] = None,
        text: Union[str, List[str]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, "TensorType"]] = None,
        **kwargs
    ) -> BatchEncoding:
        """
        Main method to prepare for the model one or several text(s) and image(s).
        """
        
        # Process images
        if images is not None:
            image_inputs = self.image_processor(images, **kwargs)
        else:
            image_inputs = {}
        
        # Process text
        if text is not None:
            text_inputs = self.tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=return_token_type_ids,
                return_length=return_length,
                verbose=verbose,
                return_tensors=return_tensors,
                **kwargs,
            )
        else:
            text_inputs = {}
        
        # Combine inputs
        return BatchEncoding({**image_inputs, **text_inputs})
    
    def decode(self, *args, **kwargs):
        """Decode token ids to strings."""
        return self.tokenizer.decode(*args, **kwargs)
    
    def batch_decode(self, *args, **kwargs):
        """Batch decode token ids to strings."""
        return self.tokenizer.batch_decode(*args, **kwargs)
    
    def post_process_generation(self, sequences, fix_markdown=False):
        """Post-process generated sequences."""
        if hasattr(self.tokenizer, 'post_process_generation'):
            return self.tokenizer.post_process_generation(sequences, fix_markdown=fix_markdown)
        else:
            # Fallback processing
            if isinstance(sequences, str):
                sequences = [sequences]
            
            processed = []
            for seq in sequences:
                # Basic cleaning
                seq = seq.replace('<s>', '').replace('</s>', '').strip()
                processed.append(seq)
            
            return processed[0] if len(processed) == 1 else processed
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load processor from pretrained model.
        
        This method is compatible with AutoProcessor.from_pretrained().
        """
        # Explicitly load subcomponents via Auto* to ensure remote auto_map is honored.
        from transformers import AutoImageProcessor, AutoTokenizer
        trust_remote_code = kwargs.get("trust_remote_code", None)
        revision = kwargs.get("revision", None)
        token = kwargs.get("token", None)
        image_processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            token=token,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            token=token,
        )
        return cls(image_processor=image_processor, tokenizer=tokenizer)
    
    def save_pretrained(self, save_directory, **kwargs):
        """
        Save processor to directory.
        
        This method is compatible with AutoProcessor/AutoImageProcessor loading.
        """
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save tokenizer with proper configuration for AutoTokenizer
        print("Saving tokenizer for AutoTokenizer compatibility...")
        self.tokenizer.save_pretrained(save_directory, **kwargs)
        
        # Save image processor  
        print("Saving image processor...")
        self.image_processor.save_pretrained(save_directory, **kwargs)
        
        # Use the parent class's save_pretrained method for processor config
        super().save_pretrained(save_directory, **kwargs)
        print(f"NemotronParseProcessor saved to {save_directory}")
        print(f"AutoTokenizer.from_pretrained('{save_directory}') should now work!")
