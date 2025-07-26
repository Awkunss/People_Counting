#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Learning based Object Tracking
Feature extraction và tracking algorithms
"""

import cv2
import numpy as np
import torch
from collections import deque
from torchvision import models, transforms
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class FeatureExtractor:
    """CNN-based feature extractor using ResNet18"""
    
    def __init__(self):
        self.model = models.resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def extract_features(self, image_patches):
        """Extract features from image patches"""
        if len(image_patches) == 0:
            return np.array([])
        
        batch = []
        for patch in image_patches:
            if patch.size > 0:
                tensor = self.transform(patch)
                batch.append(tensor)
        
        if len(batch) == 0:
            return np.array([])
        
        batch = torch.stack(batch)
        if torch.cuda.is_available():
            batch = batch.cuda()
        
        with torch.no_grad():
            features = self.model(batch)
            features = features.squeeze()
            if len(features.shape) == 1:
                features = features.unsqueeze(0)
            features = features.cpu().numpy()
        
        return features

class Track:
    """Single object track"""
    
    def __init__(self, track_id, detection, feature):
        self.track_id = track_id
        self.hits = 1
        self.time_since_update = 0
        self.feature = feature
        
        # Position tracking
        x1, y1, x2, y2 = detection[:4]
        self.x = (x1 + x2) / 2
        self.y = (y1 + y2) / 2
        self.positions = [(self.x, self.y)]
        self.bbox = detection[:4]
        
        # Crossing history for line/zone detection
        self.crossing_history = deque(maxlen=5)
        self.counted = False

    def update(self, detection, feature):
        """Update track with new detection"""
        self.hits += 1
        self.time_since_update = 0
        self.feature = 0.8 * self.feature + 0.2 * feature
        
        x1, y1, x2, y2 = detection[:4]
        self.x = (x1 + x2) / 2
        self.y = (y1 + y2) / 2
        self.positions.append((self.x, self.y))
        
        if len(self.positions) > 5:
            self.positions.pop(0)
        
        self.bbox = detection[:4]

    def predict(self):
        """Predict next state (simple implementation)"""
        self.time_since_update += 1

class DeepTracker:
    """Multi-object tracker using deep features"""
    
    def __init__(self, max_age=30, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.track_id_counter = 1
        self.feature_extractor = FeatureExtractor()

    def update(self, detections, frame):
        """Update tracker with new detections"""
        if len(detections) == 0:
            # Clean up old tracks
            self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
            for track in self.tracks:
                track.predict()
            return []
        
        # Extract features for all detections
        patches = []
        for det in detections:
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            patch = frame[y1:y2, x1:x2] 
            patches.append(patch)
        
        features = self.feature_extractor.extract_features(patches)
        cost_matrix = self._calculate_cost_matrix(detections, features)
        
        if len(self.tracks) > 0 and cost_matrix.size > 0:
            # Hungarian algorithm for optimal assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            matched_tracks = set()
            
            # Update matched tracks
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < 0.7:  # Threshold for valid match
                    self.tracks[row].update(detections[col], features[col])
                    matched_tracks.add(row)
            
            # Create new tracks for unmatched detections
            for i, det in enumerate(detections):
                if i not in col_indices or cost_matrix[row_indices[list(col_indices).index(i)], i] >= 0.7:
                    new_track = Track(self.track_id_counter, det, features[i])
                    self.tracks.append(new_track)
                    self.track_id_counter += 1
            
            # Remove old tracks
            self.tracks = [t for i, t in enumerate(self.tracks) 
                           if i in matched_tracks or t.time_since_update < self.max_age]
        else:
            # No existing tracks, create new ones
            for i, det in enumerate(detections):
                new_track = Track(self.track_id_counter, det, features[i])
                self.tracks.append(new_track)
                self.track_id_counter += 1
        
        # Return active tracks
        return [t for t in self.tracks if t.hits >= self.min_hits]

    def _calculate_cost_matrix(self, detections, features):
        """Calculate cost matrix for Hungarian algorithm"""
        if len(self.tracks) == 0 or len(detections) == 0:
            return np.array([]).reshape(0, 0)
        
        # Feature-based distances
        track_features = np.array([t.feature for t in self.tracks])
        feature_distances = cdist(track_features, features, metric='cosine')
        
        # Position-based distances
        track_positions = np.array([[t.x, t.y] for t in self.tracks])
        det_positions = np.array([[d[0] + d[2]/2, d[1] + d[3]/2] for d in detections])
        position_distances = cdist(track_positions, det_positions, metric='euclidean')
        
        # Normalize position distances
        if np.max(position_distances) > 0:
            position_distances = position_distances / np.max(position_distances)
        
        # Combine feature and position costs
        return 0.7 * feature_distances + 0.3 * position_distances
