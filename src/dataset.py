import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import h5py
import json
from filelock import FileLock

DATA_FOLDER = "mnist_gaussian_representations/"

class GaussianDataset(Dataset):
    def __init__(self, data_folder):
        self.data = []
        count = 0
        for gaussian in os.listdir(data_folder):
            file = os.path.join(data_folder, gaussian)
            data = torch.serialization.load(file, weights_only=True)['W']
            if data.shape[0] == 70:  # Ensure valid shape
                count += 1
                print(f"processing gaussian {count}...")
                self.data.append(data)
        self.data = torch.stack(self.data, dim=0)  # Final shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class GaussianDatasetSmall(Dataset):
    def __init__(self, data_folder):
        self.data = []
        count = 0
        for gaussian in os.listdir(data_folder):
            file = os.path.join(data_folder, gaussian)
            data = torch.serialization.load(file, weights_only=True)['W']
            if data.shape[0] == 70:  # Ensure valid shape
                count += 1
                print(f"processing gaussian {count}...")
                self.data.append(data)
            if count > 1001:
                break
        self.data = torch.stack(self.data, dim=0)  # Final shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class GaussianDatasetSprites(Dataset):
    def __init__(self, hdf5_filename):
        """
        Args:
            hdf5_filename (str): Path to the HDF5 file.
        """
        self.hdf5_filename = hdf5_filename
        self.data = []

        # Open the HDF5 file directly without a lock.
        with h5py.File(self.hdf5_filename, "r") as hf:
            # Iterate through group keys (e.g., "img_0", "img_1", ...) to ensure a predictable order.
            for key in hf.keys():
                # Only load the 'W' dataset from each group.
                w = hf[key]['W'][()]
                w = torch.tensor(w)
                # check the shape for the first key
                if w.shape[0] == 200:  # Ensure valid shape
                    # If the data has 9 columns, drop the 4th (index 3) column.
                    # if w.shape[1] == 9:
                    #     w = torch.cat([w[:, :3], w[:, 4:]], dim=1)
                    self.data.append(w)
        # Stack all items into a single tensor.
        self.data = torch.stack(self.data, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class GaussianDatasetSpritesBigger(Dataset):
    def __init__(self, hdf5_filename):
        """
        Args:
            hdf5_filename (str): Path to the HDF5 file.
        """
        self.hdf5_filename = hdf5_filename
        self.data = []
        self.sprite_ids = []  # Keep track of sprite IDs for reference

        with h5py.File(self.hdf5_filename, "r") as hf:
            # Print the top-level keys for debugging
            top_level_keys = list(hf.keys())
            print(f"Top-level keys in HDF5 file: {top_level_keys}")
            
            # More flexible approach - check if the structure matches any of our expectations
            if "sprite_11919" in top_level_keys or any(k.startswith("sprite_") for k in top_level_keys):
                # Case 1: Sprites are at the top level
                for sprite_id in [k for k in top_level_keys if k.startswith("sprite_")]:
                    sprite_group = hf[sprite_id]
                    if "W" in sprite_group:
                        w = torch.tensor(sprite_group["W"][()])
                        if w.shape[0] == 200:
                            self.data.append(w)
                            self.sprite_ids.append(sprite_id)
                        else:
                            print(f"Skipping {sprite_id}: unexpected shape {w.shape}")
            
            elif "W" in top_level_keys:
                # Case 2: W group contains sprites as originally expected
                w_group = hf["W"]
                print(f"Keys in W group: {list(w_group.keys())}")
                
                for sprite_id in w_group.keys():
                    sprite_group = w_group[sprite_id]
                    if "W" in sprite_group:
                        w = torch.tensor(sprite_group["W"][()])
                        if w.shape[0] == 200:
                            self.data.append(w)
                            self.sprite_ids.append(sprite_id)
                        else:
                            print(f"Skipping {sprite_id}: unexpected shape {w.shape}")
            
            # Case 3: Maybe the W datasets are directly at the top level with different naming
            elif any(k.endswith("W") for k in top_level_keys):
                for key in [k for k in top_level_keys if k.endswith("W")]:
                    w = torch.tensor(hf[key][()])
                    if w.shape[0] == 200:
                        self.data.append(w)
                        self.sprite_ids.append(key)
                    else:
                        print(f"Skipping {key}: unexpected shape {w.shape}")
            
            # Case 4: Try to find W datasets anywhere in the file
            else:
                print("Searching for W datasets throughout the file...")
                
                def find_w_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset) and name.endswith("W"):
                        try:
                            w = torch.tensor(obj[()])
                            if w.shape[0] == 200:
                                self.data.append(w)
                                self.sprite_ids.append(name)
                                print(f"Found dataset: {name} with shape {w.shape}")
                            else:
                                print(f"Skipping {name}: unexpected shape {w.shape}")
                        except Exception as e:
                            print(f"Error loading {name}: {e}")
                
                hf.visititems(find_w_datasets)

        if not self.data:
            print(f"No data loaded from {hdf5_filename}. File structure may be different than expected.")
            print("Consider using the gaussian_calc.py script to examine the file structure.")
            raise ValueError("No valid data loaded from HDF5 file.")
        
        self.data = torch.stack(self.data, dim=0)
        print(f"Successfully loaded {len(self.data)} sprites with shape {self.data.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class GaussianDatasetCIFAR10(Dataset):
    def __init__(self, hdf5_filename):
        """
        Args:
            hdf5_filename (str): Path to the HDF5 file containing CIFAR10 Gaussian representations.
        
        Returns Gaussian features in the format [200, 9] where the 9 features are:
            [sigma_x, sigma_y, rho, alpha, R, G, B, x, y]
        """
        self.hdf5_filename = hdf5_filename
        self.data = []
        
        # Open the HDF5 file to read the structure and load all data
        with h5py.File(self.hdf5_filename, "r") as hf:
            # Get all image indices from the "W" group
            if "W" in hf:
                # Sort keys to ensure consistent ordering
                image_keys = sorted([k for k in hf["W"].keys() if k.startswith("img_")], 
                                   key=lambda x: int(x.split('_')[1]))
                
                for key in image_keys:
                    img_grp = hf["W"][key]
                    
                    # Load individual parameters
                    scaling = torch.tensor(img_grp["scaling"][()])    # [num_points, 2] -> sigma_x, sigma_y
                    rotation = torch.tensor(img_grp["rotation"][()])  # [num_points, 1] -> rho
                    opacity = torch.tensor(img_grp["opacity"][()])    # [num_points, 1] -> alpha
                    features = torch.tensor(img_grp["features"][()])  # [num_points, 3] -> R, G, B
                    xyz = torch.tensor(img_grp["xyz"][()])            # [num_points, 2] -> x, y
                    
                    # Check if this has the expected number of Gaussians
                    if scaling.shape[0] != 200:
                        print(f"Skipping {key}: expected 200 gaussians, found {scaling.shape[0]}")
                        continue
                    
                    # Create tensor of shape [200, 8]
                    gaussian_features = torch.zeros((200, 8), dtype=torch.float32)
                    gaussian_features[:, 0] = scaling[:, 0]    # sigma_x
                    gaussian_features[:, 1] = scaling[:, 1]    # sigma_y
                    gaussian_features[:, 2] = rotation[:, 0]   # rho
                    # gaussian_features[:, 3] = opacity[:, 0]    # alpha
                    gaussian_features[:, 3:6] = features       # RGB
                    gaussian_features[:, 6] = xyz[:, 0]        # x
                    gaussian_features[:, 7] = xyz[:, 1]        # y
                    
                    self.data.append(gaussian_features)
                    
                print(f"Loaded {len(self.data)} images with 200 gaussians each")
                
                # Stack all items into a single tensor
                if len(self.data) > 0:
                    self.data = torch.stack(self.data, dim=0)
                else:
                    raise ValueError("No valid data found with 200 gaussians per image")
            else:
                raise ValueError("HDF5 file does not have the expected 'W' group")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class GaussianDatasetSprites64x64(Dataset):
    def __init__(self, hdf5_filename):
        """
        Args:
            hdf5_filename (str): Path to the HDF5 file containing CIFAR10 Gaussian representations.
        
        Returns Gaussian features in the format [200, 9] where the 9 features are:
            [sigma_x, sigma_y, rho, alpha, R, G, B, x, y]
        """
        self.hdf5_filename = hdf5_filename
        self.data = []
        
        # Open the HDF5 file to read the structure and load all data
        with h5py.File(self.hdf5_filename, "r") as hf:
            # Get all image indices from the "W" group
            if "W" in hf:
                # Sort keys to ensure consistent ordering
                image_keys = sorted([k for k in hf["W"].keys() if k.startswith("sprite_")], 
                                   key=lambda x: int(x.split('_')[1]))
                
                for key in image_keys:
                    img_grp = hf["W"][key]

                    
                    # Load individual parameters
                    scaling = torch.tensor(img_grp["scaling"][()])    # [num_points, 2] -> sigma_x, sigma_y
                    rotation = torch.tensor(img_grp["rotation"][()])  # [num_points, 1] -> rho
                    opacity = torch.tensor(img_grp["opacity"][()])    # [num_points, 1] -> alpha
                    features = torch.tensor(img_grp["features"][()])  # [num_points, 3] -> R, G, B
                    xyz = torch.tensor(img_grp["xyz"][()])            # [num_points, 2] -> x, y

                    
                    # Check if this has the expected number of Gaussians
                    if scaling.shape[0] != 500:
                        print(f"Skipping {key}: expected 500 gaussians, found {scaling.shape[0]}")
                        continue
                    
                    # Create tensor of shape [200, 8]
                    gaussian_features = torch.zeros((500, 8), dtype=torch.float32)
                    gaussian_features[:, 0] = scaling[:, 0]    # sigma_x
                    gaussian_features[:, 1] = scaling[:, 1]    # sigma_y
                    gaussian_features[:, 2] = rotation[:, 0]   # rho
                    # gaussian_features[:, 3] = opacity[:, 0]    # alpha
                    gaussian_features[:, 3:6] = features       # RGB
                    gaussian_features[:, 6] = xyz[:, 0]        # x
                    gaussian_features[:, 7] = xyz[:, 1]        # y
                    
                    self.data.append(gaussian_features)
                    
                print(f"Loaded {len(self.data)} images with 500 gaussians each")
                
                # Stack all items into a single tensor
                if len(self.data) > 0:
                    self.data = torch.stack(self.data, dim=0)
                else:
                    raise ValueError("No valid data found with 200 gaussians per image")
            else:
                raise ValueError("HDF5 file does not have the expected 'W' group")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    

class GaussianDatasetSprites32x32(Dataset):
    def __init__(self, hdf5_filename):
        """
        Args:
            hdf5_filename (str): Path to the HDF5 file containing CIFAR10 Gaussian representations.
        
        Returns Gaussian features in the format [200, 9] where the 9 features are:
            [sigma_x, sigma_y, rho, alpha, R, G, B, x, y]
        """
        self.hdf5_filename = hdf5_filename
        self.data = []
        
        # Open the HDF5 file to read the structure and load all data
        with h5py.File(self.hdf5_filename, "r") as hf:
            # Get all image indices from the "W" group
            if "W" in hf:
                # Sort keys to ensure consistent ordering
                image_keys = sorted([k for k in hf["W"].keys() if k.startswith("sprite_")], 
                                   key=lambda x: int(x.split('_')[1]))
                
                for key in image_keys:
                    img_grp = hf["W"][key]

                    
                    # Load individual parameters
                    scaling = torch.tensor(img_grp["scaling"][()])    # [num_points, 2] -> sigma_x, sigma_y
                    rotation = torch.tensor(img_grp["rotation"][()])  # [num_points, 1] -> rho
                    opacity = torch.tensor(img_grp["opacity"][()])    # [num_points, 1] -> alpha
                    features = torch.tensor(img_grp["features"][()])  # [num_points, 3] -> R, G, B
                    xyz = torch.tensor(img_grp["xyz"][()])            # [num_points, 2] -> x, y

                    
                    # Check if this has the expected number of Gaussians
                    if scaling.shape[0] != 150:
                        print(f"Skipping {key}: expected 500 gaussians, found {scaling.shape[0]}")
                        continue
                    
                    # Create tensor of shape [200, 8]
                    gaussian_features = torch.zeros((150, 8), dtype=torch.float32)
                    gaussian_features[:, 0] = scaling[:, 0]    # sigma_x
                    gaussian_features[:, 1] = scaling[:, 1]    # sigma_y
                    gaussian_features[:, 2] = rotation[:, 0]   # rho
                    # gaussian_features[:, 3] = opacity[:, 0]    # alpha
                    gaussian_features[:, 3:6] = features       # RGB
                    gaussian_features[:, 6] = xyz[:, 0]        # x
                    gaussian_features[:, 7] = xyz[:, 1]        # y
                    
                    self.data.append(gaussian_features)
                    
                print(f"Loaded {len(self.data)} images with 500 gaussians each")
                
                # Stack all items into a single tensor
                if len(self.data) > 0:
                    self.data = torch.stack(self.data, dim=0)
                else:
                    raise ValueError("No valid data found with 200 gaussians per image")
            else:
                raise ValueError("HDF5 file does not have the expected 'W' group")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class GaussianDatasetSprites32x32Single(Dataset):
    def __init__(self, hdf5_filename):
        """
        Loads the dataset and returns 256 copies of the first image (for sanity check).
        Each image has shape [150, 8] with features:
        [sigma_x, sigma_y, rho, R, G, B, x, y]
        """
        self.hdf5_filename = hdf5_filename
        self.data = []

        # Open the HDF5 file to read the structure and load only the first image
        with h5py.File(self.hdf5_filename, "r") as hf:
            if "W" in hf:
                image_keys = sorted(
                    [k for k in hf["W"].keys() if k.startswith("sprite_")],
                    key=lambda x: int(x.split('_')[1])
                )

                if not image_keys:
                    raise ValueError("No 'sprite_' entries found in HDF5 file.")

                # Load only the first image
                key = image_keys[0]
                img_grp = hf["W"][key]

                scaling = torch.tensor(img_grp["scaling"][()])
                rotation = torch.tensor(img_grp["rotation"][()])
                opacity = torch.tensor(img_grp["opacity"][()])
                features = torch.tensor(img_grp["features"][()])
                xyz = torch.tensor(img_grp["xyz"][()])

                if scaling.shape[0] != 150:
                    raise ValueError(f"Expected 150 gaussians, found {scaling.shape[0]}")

                # Create tensor of shape [150, 8]
                gaussian_features = torch.zeros((150, 8), dtype=torch.float32)
                gaussian_features[:, 0] = scaling[:, 0]  # sigma_x
                gaussian_features[:, 1] = scaling[:, 1]  # sigma_y
                gaussian_features[:, 2] = rotation[:, 0]  # rho
                gaussian_features[:, 3:6] = features      # RGB
                gaussian_features[:, 6] = xyz[:, 0]       # x
                gaussian_features[:, 7] = xyz[:, 1]       # y

                # Repeat the single image 256 times
                self.data = gaussian_features.unsqueeze(0).repeat(256, 1, 1)  # [256, 150, 8]

                print("Loaded 1 image and repeated it 256 times.")

            else:
                raise ValueError("HDF5 file does not have the expected 'W' group")

    def __len__(self):
        return self.data.shape[0]  # 256

    def __getitem__(self, idx):
        return self.data[idx]


class GaussianDatasetSprites32x32Subset(Dataset):
    def __init__(self, hdf5_filename, num_images=20):
        """
        Loads the first `num_images` from the HDF5 file.
        Each image has shape [150, 8] with features:
        [sigma_x, sigma_y, rho, R, G, B, x, y]
        """
        self.hdf5_filename = hdf5_filename
        self.data = []

        with h5py.File(self.hdf5_filename, "r") as hf:
            if "W" not in hf:
                raise ValueError("HDF5 file does not have the expected 'W' group.")

            image_keys = sorted(
                [k for k in hf["W"].keys() if k.startswith("sprite_")],
                key=lambda x: int(x.split('_')[1])
            )

            if not image_keys:
                raise ValueError("No 'sprite_' entries found in HDF5 file.")

            # Limit to `num_images`
            image_keys = image_keys[:num_images]

            for key in image_keys:
                img_grp = hf["W"][key]
                scaling = torch.tensor(img_grp["scaling"][()])
                rotation = torch.tensor(img_grp["rotation"][()])
                features = torch.tensor(img_grp["features"][()])
                xyz = torch.tensor(img_grp["xyz"][()])

                if scaling.shape[0] != 150:
                    print(f"Skipping {key}: expected 150 gaussians, found {scaling.shape[0]}")
                    continue

                gaussian_features = torch.zeros((150, 8), dtype=torch.float32)
                gaussian_features[:, 0] = scaling[:, 0]  # sigma_x
                gaussian_features[:, 1] = scaling[:, 1]  # sigma_y
                gaussian_features[:, 2] = rotation[:, 0]  # rho
                gaussian_features[:, 3:6] = features      # RGB
                gaussian_features[:, 6] = xyz[:, 0]       # x
                gaussian_features[:, 7] = xyz[:, 1]       # y

                self.data.append(gaussian_features)

        if not self.data:
            raise ValueError("No valid images were loaded from the HDF5 file.")

        self.data = torch.stack(self.data, dim=0)  # [num_images, 150, 8]
        print(f"Loaded {self.data.shape[0]} unique images from the dataset.")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]
