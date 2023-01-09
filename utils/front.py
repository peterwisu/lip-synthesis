#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of utility functions for facial landmark processing 

@author: Vasileios Vonikakis
"""

import numpy as np
import matplotlib.pyplot as plt
import math


def frontalize_landmarks(landmarks, frontalization_weights):
    # 
    
    '''
    ---------------------------------------------------------------------------
                      Frontalize a non-frontal face shape
    ---------------------------------------------------------------------------
    Takes an array or a list of facial landmark coordinates and returns a 
    frontalized version of them (how the face shape would look like from 
    the frontal view). Assumes 68 points with a DLIB annotation scheme. As
    described in the paper: 
    V. Vonikakis, S. Winkler. (2020). Identity Invariant Facial Landmark 
    Frontalization for Facial Expression Analysis. ICIP2020, October 2020.
    
    INPUTS
    ------
    landmarks: numpy array [68,2]
        The landmark array of the input face shape. Should follow the DLIB 
        annotation scheme.
    frontalization_weights: numpy array of [68x2+1, 68x2]
        The frontalization weights, learnt from the fill_matrices.py script. 
        The +1 is due to the interception during training. 

    OUTPUT
    ------
    landmarks_frontal: numpy array [68,2]
        The landmark array of the frontalized input face shape. 
    '''
    
    if type(landmarks) is list:
        landmarks = get_landmark_matrix(landmarks)
    
    landmarks_standard = get_procrustes(landmarks, template_landmarks=None)
    landmark_vector = np.hstack(
        (landmarks_standard[:,0].T, landmarks_standard[:,1].T, 1)
        )  # add interception
    landmarks_frontal = np.matmul(landmark_vector, frontalization_weights)
    landmarks_frontal = get_landmark_matrix(landmarks_frontal)
    
    return landmarks_frontal



def get_landmark_matrix(ls_coord):
    # Gets a list of landmark coordinates and returns a [N,2] numpy array of
    # the coordinates. Assumes that the list follows the scheme
    # [x1, x2, ..., xN, y1, y2, ..., yN]
    
    mid = len(ls_coord) // 2
    landmarks = np.array( [ ls_coord[:mid], ls_coord[mid:] ]    )
    return landmarks.T







def get_eye_centers(landmarks):
    # Given a numpy array of [68,2] facial landmarks, returns the eye centers 
    # of a face. Assumes the DLIB landmark scheme.

    landmarks_eye_left = landmarks[36:42,:]
    landmarks_eye_right = landmarks[42:48,:]
    
    center_eye_left = np.mean(landmarks_eye_left, axis=0)
    center_eye_right = np.mean(landmarks_eye_right, axis=0)
    
    return center_eye_left, center_eye_right



def get_procrustes(
        landmarks, 
        translate=True, 
        scale=True, 
        rotate=True, 
        template_landmarks=None):
    '''
    ---------------------------------------------------------------------------
                        Procrustes shape standardization
    ---------------------------------------------------------------------------
    Standardizes a given face shape, compensating for translation, scaling and
    rotation. If a template face is also given, then the standardized face is
    adjusted so as its facial parts will be displaced according to the 
    template face. More information can be found in this paper:
        
    V. Vonikakis, S. Winkler. (2020). Identity Invariant Facial Landmark 
    Frontalization for Facial Expression Analysis. ICIP2020, October 2020.
    
    INPUTS
    ------
    landmarks: numpy array [68,2]
        The landmark array of the input face shape. Should follow the DLIB 
        annotation scheme.
    translate: Boolean
        Whether or not to compensate for translation.
    scale: Boolean
        Whether or not to compensate for scaling.
    rotation: Boolean
        Whether or not to compensate for rotation.
    template_landmarks: numpy array [68,2] or None
        The landmark array of a template face shape, which will serve as 
        guidence to displace facial parts. Should follow the DLIB 
        annotation scheme. If None, no displacement is applied. 
    
    OUTPUT
    ------
    landmarks_standard: numpy array [68,2]
        The standardised landmark array of the input face shape.
        
    '''
    
    landmarks_standard = landmarks.copy()
    
    # translation
    if translate is True:
        landmark_mean = np.mean(landmarks, axis=0)
        landmarks_standard = landmarks_standard - landmark_mean
    
    # scale
    if scale is True:
        landmark_scale = math.sqrt(
            np.mean(np.sum(landmarks_standard**2, axis=1))
            )
        landmarks_standard = landmarks_standard / landmark_scale
    
    
    if rotate is True:
        # rotation
        center_eye_left, center_eye_right = get_eye_centers(landmarks_standard)
        
        # distance between the eyes
        dx = center_eye_right[0] - center_eye_left[0]
        dy = center_eye_right[1] - center_eye_left[1]
    
        if dx != 0:
            f = dy / dx
            a = math.atan(f)  # rotation angle in radians
            # ad = math.degrees(a)
            # print('Eye2eye angle=', ad)
    
        R = np.array([
            [math.cos(a), -math.sin(a)], 
            [math.sin(a), math.cos(a)]
            ])  # rotation matrix
        landmarks_standard = np.matmul(landmarks_standard, R)
    
    '''
    adjusting facial parts to a tamplate face
    displacing face parts to predetermined positions (as defined by the 
    template_landmarks), except from the eyebrows, which convey important 
    expression information attention! this only makes sense for frontal faces!
    '''
    if template_landmarks is not None:
        
        # mouth
        anchorpoint_template = np.mean(template_landmarks[50:53,:], axis=0)
        anchorpoint_input = np.mean(landmarks_standard[50:53,:], axis=0)
        displacement = anchorpoint_template - anchorpoint_input
        landmarks_standard[48:,:] += displacement
        
        # right eye
        anchorpoint_template = np.mean(template_landmarks[42:48,:], axis=0)
        anchorpoint_input = np.mean(landmarks_standard[42:48,:], axis=0)
        displacement = anchorpoint_template - anchorpoint_input
        landmarks_standard[42:48,:] += displacement
        # right eyebrow (same displaycement as the right eye)
        landmarks_standard[22:27,:] += displacement  # TODO: only X?
        
        # left eye
        anchorpoint_template = np.mean(template_landmarks[36:42,:], axis=0)
        anchorpoint_input = np.mean(landmarks_standard[36:42,:], axis=0)
        displacement = anchorpoint_template - anchorpoint_input
        landmarks_standard[36:42,:] += displacement
        # left eyebrow (same displaycement as the left eye)
        landmarks_standard[17:22,:] += displacement  # TODO: only X?
        
        # nose
        anchorpoint_template = np.mean(template_landmarks[27:36,:], axis=0)
        anchorpoint_input = np.mean(landmarks_standard[27:36,:], axis=0)
        displacement = anchorpoint_template - anchorpoint_input
        landmarks_standard[27:36,:] += displacement
        
        # jaw
        anchorpoint_template = np.mean(template_landmarks[:17,:], axis=0)
        anchorpoint_input = np.mean(landmarks_standard[:17,:], axis=0)
        displacement = anchorpoint_template - anchorpoint_input
        landmarks_standard[:17,:] += displacement
        
    return landmarks_standard



