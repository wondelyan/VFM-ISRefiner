import math
import torch
import numpy as np
import cv2
import time
import os



def get_last_point(points):
    last_point = torch.zeros((points.shape[0], 1, 4), device=points.device, dtype=points.dtype)
    last_point[:, 0, :3] = points[points[:, :, -1] == points[:, :, -1].max(dim=1)[0].unsqueeze(1)]
    last_point[:, 0, -1][
        torch.argwhere(points[:, :, -1] == points[:, :, -1].max(dim=1)[0].unsqueeze(1))[:, -1] < points.shape[
            1] // 2] = 1
    last_point[:, 0, -1][
        torch.argwhere(points[:, :, -1] == points[:, :, -1].max(dim=1)[0].unsqueeze(1))[:, -1] >= points.shape[
            1] // 2] = 0

    return last_point

def modulate_prevMask(prev_mask, points, N, R_max):
    with torch.no_grad():
        last_point = get_last_point(points)

        if torch.any(last_point < 0):
            return prev_mask

        num_points = points.shape[1] // 2
        row_array = torch.arange(start=0, end=prev_mask.shape[2], step=1, dtype=torch.float64, device=points.device)
        col_array = torch.arange(start=0, end=prev_mask.shape[3], step=1, dtype=torch.float64, device=points.device)
        coord_rows, coord_cols = torch.meshgrid(row_array, col_array)

        prevMod = prev_mask.detach().clone().to(torch.float64)
        prev_mask = prev_mask.detach().clone()

        for bindx in range(points.shape[0]):
            pos_points = points[bindx, :num_points][points[bindx, :num_points, -1] != -1]
            neg_points = points[bindx, num_points:][points[bindx, num_points:, -1] != -1]

            y, x = last_point[bindx, 0, :2]
            p = prev_mask[bindx, 0, y.long(), x.long()]  # Click point probability value

            # Calculate the initial circular window
            dist = torch.sqrt((coord_rows - y) ** 2 + (coord_cols - x) ** 2)
            if last_point[bindx, :, -1] == 1:  # Positive click
                if neg_points.shape[0] != 0:
                    min_dist = torch.cdist(neg_points[:, :2], last_point[bindx, 0, :2].unsqueeze(0)).min(dim=0)[0]
                    r = min_dist / 2
                    modWindow = (dist <= r)
                    if r < 10:
                        r = 10
                        modWindow = (dist <= r)
                        if min_dist < 10:
                            in_modWindow = neg_points[
                                (torch.cdist(neg_points[:, :2], last_point[bindx, 0, :2].unsqueeze(0)) < 10)[:, 0]]
                            for n_click in in_modWindow:
                                dist_n = torch.sqrt((coord_rows - n_click[0])** 2 + (coord_cols - n_click[1])** 2)
                                modWindow_n = (dist_n <= torch.sqrt((last_point[bindx, 0, 0] - n_click[0])** 2 + 
                                                           (last_point[bindx, 0, 1] - n_click[1])** 2))
                                modWindow[modWindow_n] = 0
                else:
                    r = R_max
                    modWindow = (dist <= r)
                
                # Extract probability values within the initial window
                window_probs = prev_mask[bindx, 0, modWindow]
                if window_probs.numel() == 0:
                    continue
                
                # Calculate probability statistics
                window_max = torch.max(window_probs).item()
                window_mean = torch.mean(window_probs).item()
                window_median = torch.median(window_probs).item()
                
                # Compare the probability of clicking points with statistics to find the minimum value
                compare_values = [p, window_mean, window_median]
                min_val = min(compare_values)
                
                # Initialize new_modWindow
                new_modWindow = torch.zeros_like(prev_mask[bindx, 0], dtype=torch.bool)

                # Extract prob_map from modWindow
                prob_map = prev_mask[bindx, 0][modWindow]

                # Generate a Boolean mask that satisfies the range of [min_mal, window_max]
                condition = (prob_map >= min_val) & (prob_map <= window_max)

                # Update new_modWindow
                new_modWindow[modWindow] = condition
                
                # Select the maximum gamma value
                if p == 0:
                    prevMod[bindx, 0][new_modWindow] = 1 - (dist[new_modWindow] / dist[new_modWindow].max())
                    continue
                elif p < 0.99:
                    max_gamma = 1 / (math.log(0.99, p) + 1e-8)
                else:
                    max_gamma = 1
                dist_new = dist[new_modWindow]
                exp = max_gamma * (1 - (dist_new / r)) + (dist_new / r)
                # Modulate the previous round mask
                prevMod[bindx, 0][new_modWindow] = prevMod[bindx, 0][new_modWindow] ** (1 / exp)
                prevMod[bindx, 0][int(y.round()), int(x.round())] = 1
            else: # Negative click
                if pos_points.shape[0] != 0:
                    min_dist = torch.cdist(pos_points[:, :2], last_point[bindx, 0, :2].unsqueeze(0)).min(dim=0)[0]
                    r = min_dist / 2
                    modWindow = (dist <= r)
                    if r < 10:
                        r = 10
                        modWindow = (dist <= r)
                        if min_dist < 10:
                            in_modWindow = pos_points[
                                (torch.cdist(pos_points[:, :2], last_point[bindx, 0, :2].unsqueeze(0)) < 10)[:, 0]]
                            for n_click in in_modWindow:
                                dist_n = torch.sqrt((coord_rows - n_click[0])** 2 + (coord_cols - n_click[1])** 2)
                                modWindow_n = (dist_n <= torch.sqrt((last_point[bindx, 0, 0] - n_click[0])** 2 + 
                                                           (last_point[bindx, 0, 1] - n_click[1])** 2))
                                modWindow[modWindow_n] = 0
                else:
                    r = R_max
                    modWindow = (dist <= r)
                
                window_probs = prev_mask[bindx, 0, modWindow]
                if window_probs.numel() == 0:
                    continue
                window_min = window_probs.min().item()
                window_mean = window_probs.mean().item()
                window_median = torch.median(window_probs).item()
                
                compare_values = [p, window_mean, window_median]
                max_val = max(compare_values)
                
                new_modWindow = torch.zeros_like(prev_mask[bindx, 0], dtype=torch.bool)

                prob_map = prev_mask[bindx, 0][modWindow]
                
                condition = (prob_map <= max_val) & (prob_map >= window_min)

                new_modWindow[modWindow] = condition
                
                if p == 1:
                    prevMod[bindx, 0][new_modWindow] = dist[new_modWindow] / dist[new_modWindow].max()
                    continue
                elif p > 0.01:
                    max_gamma = math.log(0.01, p)
                else:
                    max_gamma = 1
            
                dist_new = dist[new_modWindow]
                exp = max_gamma * (1 - (dist_new / r)) + (dist_new / r)

                prevMod[bindx, 0][new_modWindow] = prevMod[bindx, 0][new_modWindow] ** (exp)
                prevMod[bindx, 0][int(y.round()), int(x.round())] = 0       
    return prevMod.to(torch.float32)
