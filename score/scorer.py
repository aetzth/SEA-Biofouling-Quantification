"""
Indirect Biofouling Severity Quantification Engine for Offshore Net Cages
Core algorithmic implementation corresponding to Section 2.2 of the manuscript.

Key Steps:
1. Zhang-Suen Thinning & Connected Component Analysis (Mesh Skeletonization)
2. Geometrical Template Matching with Fallback for Adaptive Threshold T = r_clean + delta
3. Aperture-level Blockage Proportion Calculation & Boundary Propagation
4. Severity Aggregation and Dimensionless Normalization (0 - 100)
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from PIL import Image


# ============================================================================
# 1. Image I/O Utilities
# ============================================================================

def robust_read_image(img_path: Union[str, Path], flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Read image robustly across platforms, returning BGR ndarray."""
    path_str = str(img_path)
    if not os.path.exists(path_str):
        return None
    try:
        data = np.fromfile(path_str, dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        if img is not None:
            return img
    except Exception:
        pass

    try:
        pil_img = Image.open(path_str)
        if flags == cv2.IMREAD_GRAYSCALE:
            return np.array(pil_img.convert('L'))
        else:
            rgb = np.array(pil_img.convert('RGB'))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def robust_read_mask(mask_path: Union[str, Path], target_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """
    Read binary segmentation mask.
    Returns uint8 ndarray [H, W] where 1 denotes clean open aperture, 0 denotes twine/fouling.
    """
    img = robust_read_image(mask_path, flags=cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    if target_size is not None and (img.shape[0], img.shape[1]) != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)

    mask = (img > 0).astype(np.uint8)
    return mask


def robust_write_image(save_path: Union[str, Path], img: np.ndarray) -> bool:
    """Save image array to file path robustly."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ext = save_path.suffix
    try:
        ok, buf = cv2.imencode(ext, img)
        if ok:
            buf.tofile(str(save_path))
            return True
    except Exception:
        pass

    try:
        if len(img.shape) == 2:
            pil_img = Image.fromarray(img)
        else:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil_img.save(str(save_path))
        return True
    except Exception:
        return False


# ============================================================================
# 2. Skeleton Extraction & Connected Component Analysis
# ============================================================================

def mask_to_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """
    Extract 1-pixel wide netting twine skeleton from aperture binary mask.
    Inverts mask (1=aperture, 0=twine/fouling) and applies Zhang-Suen thinning.
    """
    inverted = (binary_mask == 0)
    skel_bool = skeletonize(inverted)
    return (skel_bool * 255).astype(np.uint8)


def extract_valid_regions(skeleton_img: np.ndarray, border_threshold: int = 5) -> List[Dict[str, Any]]:
    """
    Extract non-border enclosed aperture regions from skeletonized netting image.
    """
    if skeleton_img is None:
        return []

    img_height, img_width = skeleton_img.shape
    black_regions = (skeleton_img == 0).astype(np.uint8)
    labeled_array, num_features = ndimage.label(black_regions)

    valid_regions = []
    for label_id in range(1, num_features + 1):
        component_mask = (labeled_array == label_id).astype(np.uint8)
        area = int(np.sum(component_mask))
        if area <= 0:
            continue

        y_coords, x_coords = np.where(component_mask == 1)
        min_dist_to_border = min(
            int(np.min(y_coords)),
            int(np.min(x_coords)),
            int(img_height - 1 - np.max(y_coords)),
            int(img_width - 1 - np.max(x_coords))
        )

        if min_dist_to_border <= border_threshold:
            continue

        contours, _ = cv2.findContours(component_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(contour, True))

        min_area_rect = cv2.minAreaRect(contour)
        rect_w, rect_h = min_area_rect[1]
        if min(rect_w, rect_h) > 0:
            aspect_ratio = float(max(rect_w, rect_h) / min(rect_w, rect_h))
        else:
            aspect_ratio = float('inf')

        _, _, bw, bh = cv2.boundingRect(contour)
        rect_area = bw * bh
        rectangularity = float(area / rect_area) if rect_area > 0 else 0.0

        valid_regions.append({
            'label_id': label_id,
            'area': area,
            'perimeter': perimeter,
            'contour': contour,
            'mask': component_mask,
            'aspect_ratio': aspect_ratio,
            'rectangularity': rectangularity,
            'min_rect_size': (rect_w, rect_h),
            'min_dist_to_border': min_dist_to_border
        })

    return valid_regions


def find_regions_sharing_skeleton_boundary(labeled_array: np.ndarray,
                                           skeleton_img: np.ndarray,
                                           region_id: int) -> np.ndarray:
    """Find adjacent aperture component IDs sharing skeleton boundary lines."""
    region_mask = (labeled_array == region_id)
    region_dilated = ndimage.binary_dilation(region_mask)
    region_boundary = region_dilated & ~region_mask

    skeleton_boundary = (skeleton_img == 255) & region_boundary
    skeleton_boundary_dilated = ndimage.binary_dilation(skeleton_boundary)

    adjacent_labels = np.unique(labeled_array[skeleton_boundary_dilated])
    adjacent_labels = adjacent_labels[(adjacent_labels != 0) & (adjacent_labels != region_id)]
    return adjacent_labels


def is_region_touching_border(region_mask: np.ndarray) -> bool:
    """Check whether a connected region intersects image boundaries."""
    if np.any(region_mask[0, :]) or np.any(region_mask[-1, :]):
        return True
    if np.any(region_mask[:, 0]) or np.any(region_mask[:, -1]):
        return True
    return False


# ============================================================================
# 3. Biofouling Scorer Class
# ============================================================================

class BiofoulingScorer:
    """
    Indirect Aperture-Structure Biofouling Severity Quantification Engine.
    Implements Equations (2) to (6) described in Section 2.2 of the manuscript.
    """

    def __init__(self,
                 templates_dir: Optional[Union[str, Path]] = None,
                 border_threshold: int = 5,
                 max_aspect_ratio: float = 1.2,
                 w1: float = 0.6,
                 w2: float = 0.4):
        """
        Initialize BiofoulingScorer.

        Args:
            templates_dir: Path to directory containing templates (binary, skeleton, template_metadata.json).
            border_threshold: Border margin threshold in pixels.
            max_aspect_ratio: Maximum aspect ratio threshold for selecting regular candidate apertures.
            w1: Weight for area relative difference (default: 0.6).
            w2: Weight for perimeter relative difference (default: 0.4).
        """
        if templates_dir is None:
            current_dir = Path(__file__).resolve().parent
            templates_dir = current_dir / "templates"

        self.templates_dir = Path(templates_dir)
        self.binary_dir = self.templates_dir / "binary"
        self.skeleton_dir = self.templates_dir / "skeleton"
        self.meta_path = self.templates_dir / "template_metadata.json"

        self.border_threshold = border_threshold
        self.max_aspect_ratio = max_aspect_ratio
        self.w1 = w1
        self.w2 = w2

        self.metadata = self._load_metadata()
        self.template_cache = self._preload_templates()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load template metadata configuration."""
        if self.meta_path.exists():
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "templates": {
                "c_01.png": {"base_proportion": 0.582251, "safety_margin": 0.38, "threshold": 0.962251},
                "c_02.png": {"base_proportion": 0.417854, "safety_margin": 0.32, "threshold": 0.737854},
                "c_03.png": {"base_proportion": 0.280816, "safety_margin": 0.29, "threshold": 0.570816},
                "c_04.png": {"base_proportion": 0.304476, "safety_margin": 0.26, "threshold": 0.564476},
                "c_05.png": {"base_proportion": 0.318232, "safety_margin": 0.22, "threshold": 0.538232},
                "c_06.png": {"base_proportion": 0.323467, "safety_margin": 0.20, "threshold": 0.523467},
                "c_07.png": {"base_proportion": 0.272358, "safety_margin": 0.18, "threshold": 0.452358},
                "c_08.png": {"base_proportion": 0.292278, "safety_margin": 0.14, "threshold": 0.432278},
            }
        }

    def _preload_templates(self) -> Dict[str, Dict[str, Any]]:
        """Pre-compute geometric features of clean templates to avoid repeated I/O."""
        cache = {}
        for tpl_name, tpl_info in self.metadata.get("templates", {}).items():
            skel_file = self.skeleton_dir / tpl_name
            skel_img = robust_read_image(skel_file, flags=cv2.IMREAD_GRAYSCALE)
            if skel_img is None:
                continue

            regions = extract_valid_regions(skel_img, border_threshold=self.border_threshold)
            cache[tpl_name] = {
                'info': tpl_info,
                'regions': regions,
                'base_proportion': tpl_info.get("base_proportion", 0.3),
                'safety_margin': tpl_info.get("safety_margin", 0.2),
                'threshold': tpl_info.get("threshold", 0.5)
            }
        return cache

    def find_best_matching_template(self, skeleton_img: np.ndarray) -> Tuple[Optional[str], float, Optional[Dict[str, Any]]]:
        """
        Find closest clean template matching the current netting scale.
        Implements Equation (2) with fallback mechanism.
        """
        regions = extract_valid_regions(skeleton_img, border_threshold=self.border_threshold)
        if not regions:
            default_tpl = "c_05.png" if "c_05.png" in self.template_cache else list(self.template_cache.keys())[0]
            return default_tpl, 0.5, None

        regions_sorted = sorted(regions, key=lambda x: x['rectangularity'], reverse=True)

        candidate = None
        for comp in regions_sorted:
            if comp['aspect_ratio'] <= self.max_aspect_ratio:
                candidate = comp
                break

        if candidate is None:
            candidate = min(regions, key=lambda x: x['aspect_ratio'])

        target_area = candidate['area']
        target_perimeter = candidate['perimeter']

        best_match = None
        min_dist = float('inf')

        for tpl_name, tpl_data in self.template_cache.items():
            tpl_regions = tpl_data['regions']
            if not tpl_regions:
                continue

            for t_reg in tpl_regions:
                t_area = t_reg['area']
                t_perim = t_reg['perimeter']
                if t_area <= 0 or t_perim <= 0:
                    continue

                area_diff = abs(target_area / t_area - 1.0)
                perim_diff = abs(target_perimeter / t_perim - 1.0)
                dist = self.w1 * area_diff + self.w2 * perim_diff

                if dist < min_dist:
                    min_dist = dist
                    best_match = tpl_name

        if best_match is None:
            best_match = "c_05.png"
            min_dist = 0.5

        return best_match, min_dist, candidate

    def calculate_proportions(self, binary_mask: np.ndarray, skeleton_img: np.ndarray) -> Tuple[Dict[int, float], np.ndarray]:
        """
        Calculate local aperture blockage proportions B_k (Eq. 4).
        Applies topological boundary inference for partial border apertures.
        """
        black_regions = (skeleton_img == 0).astype(np.uint8)
        labeled_array, num_features = ndimage.label(black_regions)

        proportions = {}
        border_regions = set()
        non_border_proportions = {}
        all_regions_are_border = True

        for region_id in range(1, num_features + 1):
            region_mask = (labeled_array == region_id)
            if is_region_touching_border(region_mask):
                border_regions.add(region_id)
            else:
                all_regions_are_border = False
                region_in_binary = binary_mask[region_mask]
                black_pixels = np.sum(region_in_binary == 0)
                total_pixels = region_in_binary.size
                p = float(black_pixels / total_pixels) if total_pixels > 0 else 0.0
                non_border_proportions[region_id] = p
                proportions[region_id] = p

        if all_regions_are_border:
            for region_id in range(1, num_features + 1):
                region_mask = (labeled_array == region_id)
                region_in_binary = binary_mask[region_mask]
                black_pixels = np.sum(region_in_binary == 0)
                total_pixels = region_in_binary.size
                p = float(black_pixels / total_pixels) if total_pixels > 0 else 0.0
                proportions[region_id] = p
            return proportions, labeled_array

        def get_region_area(r_id):
            return np.sum(labeled_array == r_id)

        processed_border_regions = set()
        remaining_border_regions = border_regions.copy()

        while remaining_border_regions:
            regions_to_process = set()
            for border_id in remaining_border_regions:
                adj_labels = find_regions_sharing_skeleton_boundary(labeled_array, skeleton_img, border_id)
                has_valid = any(adj in non_border_proportions or adj in processed_border_regions for adj in adj_labels)
                if has_valid:
                    regions_to_process.add(border_id)

            if not regions_to_process:
                regions_to_process = remaining_border_regions.copy()

            for border_id in regions_to_process:
                adj_labels = find_regions_sharing_skeleton_boundary(labeled_array, skeleton_img, border_id)
                valid_adjs = [adj for adj in adj_labels if adj in non_border_proportions or adj in processed_border_regions]

                if valid_adjs:
                    largest_adj = max(valid_adjs, key=get_region_area)
                    p_inherited = non_border_proportions.get(largest_adj, proportions.get(largest_adj, 0.0))
                    proportions[border_id] = p_inherited
                else:
                    region_mask = (labeled_array == border_id)
                    region_in_binary = binary_mask[region_mask]
                    black_pixels = np.sum(region_in_binary == 0)
                    total_pixels = region_in_binary.size
                    p_raw = float(black_pixels / total_pixels) if total_pixels > 0 else 0.0
                    proportions[border_id] = p_raw

                processed_border_regions.add(border_id)
                remaining_border_regions.remove(border_id)

        return proportions, labeled_array

    def score_mask(self,
                   binary_mask: np.ndarray,
                   skeleton_img: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        End-to-end memory-level scoring for an aperture binary mask.
        Computes S_raw (Eq. 5) and S_norm (Eq. 6).
        """
        h, w = binary_mask.shape[:2]

        if skeleton_img is None:
            skeleton_img = mask_to_skeleton(binary_mask)

        matched_tpl, dist, cand_info = self.find_best_matching_template(skeleton_img)

        tpl_data = self.template_cache.get(matched_tpl, {})
        base_thresh = tpl_data.get('base_proportion', 0.3)
        safety_margin = tpl_data.get('safety_margin', 0.2)
        threshold = base_thresh + safety_margin

        proportions, labeled_array = self.calculate_proportions(binary_mask, skeleton_img)

        raw_score = 0.0
        num_blocked_apertures = 0
        for region_id, p_k in proportions.items():
            if p_k > threshold:
                num_blocked_apertures += 1
                region_area = float(np.sum(labeled_array == region_id))
                raw_score += region_area * (p_k - threshold)

        max_score = float(h * w * (1.0 - threshold))
        if max_score > 0:
            norm_score = float((raw_score / max_score) * 100.0)
        else:
            norm_score = 0.0

        norm_score = max(0.0, min(100.0, norm_score))

        return {
            'raw_score': round(raw_score, 2),
            'normalized_score': round(norm_score, 2),
            'threshold': round(threshold, 6),
            'base_threshold': round(base_thresh, 6),
            'safety_margin': round(safety_margin, 4),
            'matched_template': matched_tpl,
            'distance_to_template': round(dist, 4),
            'max_score': round(max_score, 2),
            'num_total_apertures': len(proportions),
            'num_fouled_apertures': num_blocked_apertures,
            'proportions': proportions,
            'labeled_array': labeled_array,
            'skeleton_img': skeleton_img
        }

    def score_mask_file(self,
                        mask_path: Union[str, Path],
                        skeleton_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Compute score from file path."""
        mask = robust_read_mask(mask_path)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask image from: {mask_path}")

        skeleton_img = None
        if skeleton_path is not None and os.path.exists(str(skeleton_path)):
            skeleton_img = robust_read_image(skeleton_path, flags=cv2.IMREAD_GRAYSCALE)

        return self.score_mask(mask, skeleton_img)


if __name__ == "__main__":
    scorer = BiofoulingScorer()
    print(f"BiofoulingScorer initialized successfully with {len(scorer.template_cache)} templates.")
    for name, data in scorer.template_cache.items():
        print(f"  - {name}: r_clean={data['base_proportion']:.4f}, margin={data['safety_margin']:.2f}, T={data['threshold']:.4f}")
