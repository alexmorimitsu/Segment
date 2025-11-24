import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QFrame,
                             QSlider, QCheckBox, QRadioButton, QButtonGroup, QScrollArea,
                             QMessageBox, QProgressDialog)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QKeySequence, QImage, QBrush
from PyQt5.QtCore import Qt, QSize, QPoint
from PyQt5.QtWidgets import QShortcut
import random


class Segmentation:
    """Class to represent a segmented region"""
    
    def __init__(self, seg_id, color, pixels):
        self.id = seg_id
        self.color = color  # tuple of 3 ints (R, G, B)
        self.pixels = pixels  # list of pairs of ints (x, y)
        self.border_pixels = []  # list of border pixels (x, y)
    
    def add_pixels(self, new_pixels):
        """Add new pixels to the segmentation"""
        self.pixels.extend(new_pixels)
    
    def get_pixel_count(self):
        """Get the number of pixels in this segmentation"""
        return len(self.pixels)
    
    def get_border_count(self):
        """Get the number of border pixels in this segmentation"""
        return len(self.border_pixels)


class Drawing:
    """Class to represent a drawn line"""
    
    def __init__(self, draw_id, pixels, thickness=1):
        self.id = draw_id
        self.pixels = pixels  # list of pairs of ints (x, y)
        self.thickness = thickness  # Pen thickness used for this drawing
    
    def add_pixels(self, new_pixels):
        """Add new pixels to the drawing"""
        self.pixels.extend(new_pixels)
    
    def get_pixel_count(self):
        """Get the number of pixels in this drawing"""
        return len(self.pixels)


class ImageCanvas(QLabel):
    """Custom canvas widget for displaying images"""
    
    def __init__(self):
        super().__init__()
        self.setFixedSize(1280, 720)  # Set fixed 1280x720 canvas size
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #cccccc;
                background-color: #f0f0f0;
                border-radius: 5px;
            }
        """)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Nenhuma imagem carregada")
        self.original_pixmap = None
        self.setMouseTracking(True)  # Enable mouse tracking for drawing
        self.setFocusPolicy(Qt.StrongFocus)  # Enable keyboard focus for key events
        self.setToolTip("Selecionar: Clique para selecionar uma região\nSegurar Shift + Clique para selecionar múltiplas regiões\nRemover: Clique em uma região para removê-la completamente")
        self.mask_pixmap = None
        self.mask_opacity = 0.5  # Opacity for mask overlay (0.0 to 1.0)
        # Cache for scaled mask to avoid redrawing
        self.scaled_mask_cache = None
        self.scaled_mask_cache_zoom = None
        self.scaled_mask_cache_size = None
        self.mode = "none"  # "none", "drawing", "drag", "paint", "select", or "remove"
        self.last_pos = None
        self.drawing = False
        self.zoom_factor = 1
        self.min_zoom = 1
        self.max_zoom = 3
        self.max_image_width = 1280
        self.max_image_height = 720
        self.viewport_x = 0
        self.viewport_y = 0
        self.segmentations = {}  # Dictionary to store Segmentation objects (id -> Segmentation)
        self.next_seg_id = 1  # Counter for segmentation IDs
        self.drawings = {}  # Dictionary to store Drawing objects (id -> Drawing)
        self.next_draw_id = 1  # Counter for drawing IDs
        self.current_drawing = None  # Current drawing being created
        self.pen_thickness = 3  # Pen thickness for drawing
        self.undo_stack = []  # Stack to store operations for undo
        self.redo_stack = []  # Stack to store operations for redo
        self.changes_made = False  # Track if any changes have been made
        self.pixel_labels = None  # Array mapping each pixel to its segmentation label (0 = no label)
        self.selected_segmentations = set()  # Set of selected segmentation IDs
        self.selection_overlay = None  # Overlay for showing selected regions in red
        self.scaled_selection_cache = None  # Cache for scaled selection overlay
        self.scaled_selection_cache_zoom = None  # Zoom factor when cache was created
        self.scaled_selection_cache_size = None  # Size when cache was created
        self.right_panel = None  # Will be set by set_right_panel method
        self.main_window = None  # Will be set by set_main_window method
    
    def set_right_panel(self, right_panel):
        """Set the right panel reference"""
        self.right_panel = right_panel
    
    def set_main_window(self, main_window):
        """Set the main window reference"""
        self.main_window = main_window
        
    def load_image(self, image_path):
        """Load and display an image on the canvas"""
        if os.path.exists(image_path):
            # Lock interface during loading
            self.lock_interface("Carregando imagem...")
            
            try:
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    if (pixmap.width() > self.max_image_width or 
                        pixmap.height() > self.max_image_height):
                        pixmap = pixmap.scaled(
                            QSize(self.max_image_width, self.max_image_height),
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation
                        )
                    self.original_pixmap = pixmap
                    # Initialize pixel labels array
                    self.initialize_pixel_labels()
                    self.display_image()
                    self.setText("")
                else:
                    self.setText("Failed to load image")
            finally:
                self.unlock_interface()
        else:
            self.setText("Image file not found")
    
    def load_mask(self, mask_path):
        """Load a mask from file and compute border cache"""
        # Lock interface during loading
        self.lock_interface("Carregando máscara...")
        
        try:
            # Clear existing segmentations when loading a new mask
            self.segmentations = {}
            self.next_seg_id = 1
            self.drawings = {}
            self.next_draw_id = 1
            self.undo_stack = []
            self.redo_stack = []
            
            # Update undo/redo button states
            self.update_undo_redo_buttons()
            
            # Reset changes tracking
            self.changes_made = False
            
            # Reset pixel labels array
            self.pixel_labels = None
            
            if os.path.exists(mask_path):
                mask_pixmap = QPixmap(mask_path)
                if not mask_pixmap.isNull():
                    self.mask_pixmap = mask_pixmap
                    print(f"Mask loaded from: {mask_path}")
                    
                    # Invalidate cache when mask changes
                    self.invalidate_scaled_mask_cache()
                    
                    # Try to load labels array from corresponding .npy file
                    base_path = os.path.splitext(mask_path)[0]  # Remove extension
                    labels_path = f"{base_path}_labels.npy"
                    
                    if os.path.exists(labels_path):
                        try:
                            # Load the labels array
                            labels_array = np.load(labels_path)
                            print(f"Labels array loaded from: {labels_path}")
                            
                            # Extract segmentations from the labels array
                            self.extract_segmentations_from_labels(labels_array)
                        except Exception as e:
                            print(f"Failed to load labels array: {e}")
                            # Fall back to extracting from mask image
                            self.extract_segmentations_from_mask()
                    else:
                        print(f"Labels file not found: {labels_path}, extracting from mask image")
                        # Extract segmentations from the mask image
                        self.extract_segmentations_from_mask()
                    
                    # Update the right panel to show segmentations (even if empty)
                    self.update_segmentation_display()
                    
                    self.display_image()
                else:
                    print(f"Failed to load mask: {mask_path}")
                    self.create_blank_mask()
            else:
                print(f"Mask file not found: {mask_path}")
                self.create_blank_mask()
        finally:
            self.unlock_interface()
    
    def create_blank_mask(self):
        """Create a blank mask for drawing"""
        if self.original_pixmap:
            # Clear existing segmentations when creating a new blank mask
            self.segmentations = {}
            self.next_seg_id = 1
            self.drawings = {}
            self.next_draw_id = 1
            self.undo_stack = []
            self.redo_stack = []
            
            # Update undo/redo button states
            self.update_undo_redo_buttons()
            
            # Reset pixel labels array
            self.pixel_labels = None
            
            # Reset changes tracking
            self.changes_made = False
            
            self.mask_pixmap = QPixmap(self.original_pixmap.size())
            self.mask_pixmap.fill(QColor(0, 0, 0))  # Black background
            # Invalidate cache when mask changes
            self.invalidate_scaled_mask_cache()
            self.display_image()
            
            # Update the right panel to show empty segmentations
            self.update_segmentation_display()
            
            # Notify main window that a blank mask was created
            if self.main_window:
                self.main_window.current_mask_path = None
                self.main_window.update_window_title()
    
    def extract_segmentations_from_mask(self):
        """Extract segmentations from the mask image by analyzing colored pixels"""
        if not self.mask_pixmap:
            return
        
        print("Extracting segmentations from mask image...")
        
        # Convert mask to image for pixel analysis
        mask_image = self.mask_pixmap.toImage()
        width, height = mask_image.width(), mask_image.height()
        
        # Dictionary to store color -> pixel coordinates
        color_to_pixels = {}
        
        # Analyze all pixels in the mask
        for y in range(height):
            for x in range(width):
                color = mask_image.pixelColor(x, y)
                rgb = (color.red(), color.green(), color.blue())
                
                # Skip black pixels (background) and white pixels (borders)
                if rgb != (0, 0, 0) and rgb != (255, 255, 255):
                    if rgb not in color_to_pixels:
                        color_to_pixels[rgb] = []
                    color_to_pixels[rgb].append((x, y))
        
        # Create segmentations for each color found
        for color_rgb, pixels in color_to_pixels.items():
            if len(pixels) > 0:  # Only create segmentation if there are pixels
                segmentation = Segmentation(self.next_seg_id, color_rgb, pixels)
                self.segmentations[self.next_seg_id] = segmentation
                self.next_seg_id += 1
                
                print(f"Created segmentation {segmentation.id} with {len(pixels)} pixels, color {color_rgb}")
        
        # If we don't have an original pixmap, use the mask dimensions
        if not self.original_pixmap:
            self.original_pixmap = self.mask_pixmap
        
        # Update pixel labels array
        self.update_pixel_labels()
        
        # Compute borders for all extracted segmentations
        for segmentation in self.segmentations.values():
            self.compute_segmentation_borders(segmentation)
        
        print(f"Extracted {len(self.segmentations)} segmentations from mask image")
    
    def extract_segmentations_from_labels(self, labels_array):
        """Extract segmentations from the labels array using the original IDs"""
        if labels_array is None:
            return
        
        print("Extracting segmentations from labels array...")
        
        height, width = labels_array.shape
        
        # Dictionary to store label -> pixel coordinates
        label_to_pixels = {}
        
        # Analyze all pixels in the labels array
        for y in range(height):
            for x in range(width):
                label = labels_array[y, x]
                
                # Skip background (label 0)
                if label > 0:
                    if label not in label_to_pixels:
                        label_to_pixels[label] = []
                    label_to_pixels[label].append((x, y))
        
        # Create segmentations for each label found, preserving original IDs
        for label, pixels in label_to_pixels.items():
            if len(pixels) > 0:  # Only create segmentation if there are pixels
                # Generate a random color for this segmentation
                random_color = self.generate_random_color()
                color_tuple = (random_color.red(), random_color.green(), random_color.blue())
                
                # Create segmentation with the original label ID
                segmentation = Segmentation(label, color_tuple, pixels)
                self.segmentations[label] = segmentation
                
                # Update next_seg_id to be higher than any existing ID
                if label >= self.next_seg_id:
                    self.next_seg_id = label + 1
                
                print(f"Created segmentation {segmentation.id} with {len(pixels)} pixels")
        
        # If we don't have an original pixmap, create one with the mask dimensions
        if not self.original_pixmap:
            self.original_pixmap = QPixmap(width, height)
        
        # Set the pixel labels array directly from the loaded array
        self.pixel_labels = labels_array.astype(np.int32)
        
        # Compute borders for all extracted segmentations
        for segmentation in self.segmentations.values():
            self.compute_segmentation_borders(segmentation)
        
        print(f"Extracted {len(self.segmentations)} segmentations from labels array")
    
    def redraw_all_segmentations(self):
        """Redraw all segmentations on the mask using NumPy for efficiency"""
        if not self.mask_pixmap:
            return
        
        # Calculate total pixels to determine if we need progress
        total_pixels = sum(len(seg.pixels) for seg in self.segmentations.values())
        total_segmentations = len(self.segmentations)
        
        # Only show progress for large operations (more than 5000 pixels or more than 10 segmentations)
        progress = None
        if total_pixels > 5000 or total_segmentations > 10:
            progress = self.show_progress_dialog("Redesenhando Segmentações", f"Processando {total_segmentations} segmentações...", total_segmentations)
        
        try:
            # Convert pixmap to image for pixel manipulation
            image = self.mask_pixmap.toImage()
            width, height = image.width(), image.height()
            
            # Create a NumPy array for efficient batch operations
            # First, clear the image to black
            self._clear_image_to_color(image, QColor(0, 0, 0))
            
            # Redraw all segmentation pixels with their original colors using NumPy
            processed_segmentations = 0
            for segmentation in self.segmentations.values():
                # Convert segmentation pixels to NumPy array for batch operations
                if segmentation.pixels:
                    pixels_array = np.array(segmentation.pixels)
                    
                    # Filter valid pixels (within bounds)
                    valid_mask = ((pixels_array[:, 0] >= 0) & (pixels_array[:, 0] < width) & 
                                 (pixels_array[:, 1] >= 0) & (pixels_array[:, 1] < height))
                    valid_pixels = pixels_array[valid_mask]
                    
                    # Set all pixels for this segmentation at once
                    for x, y in valid_pixels:
                        image.setPixelColor(int(x), int(y), QColor(*segmentation.color))
                
                # Update progress only if progress dialog exists
                processed_segmentations += 1
                if progress:
                    progress.setValue(processed_segmentations)
                    if progress.wasCanceled():
                        break
            
            # Convert back to pixmap
            self.mask_pixmap = QPixmap.fromImage(image)
            
            # Compute borders for all segmentations
            if progress:
                progress.setLabelText("Calculando bordas...")
                progress.setValue(0)
                progress.setMaximum(len(self.segmentations))
            
            processed_segmentations = 0
            for segmentation in self.segmentations.values():
                self.compute_segmentation_borders(segmentation)
                processed_segmentations += 1
                if progress:
                    progress.setValue(processed_segmentations)
                    if progress.wasCanceled():
                        break
            
            print(f"Computed borders for {len(self.segmentations)} segmentations")
            
            # Invalidate cache when mask changes
            self.invalidate_scaled_mask_cache()
        finally:
            if progress:
                progress.close()
    
    def redraw_single_segmentation(self, segmentation):
        """Redraw a single segmentation on the mask without reprocessing borders"""
        if not self.mask_pixmap or not segmentation:
            return
        
        # Convert pixmap to image for pixel manipulation
        image = self.mask_pixmap.toImage()
        width, height = image.width(), image.height()
        
        # Redraw only this segmentation's pixels with its color
        if segmentation.pixels:
            pixels_array = np.array(segmentation.pixels)
            
            # Filter valid pixels (within bounds)
            valid_mask = ((pixels_array[:, 0] >= 0) & (pixels_array[:, 0] < width) & 
                         (pixels_array[:, 1] >= 0) & (pixels_array[:, 1] < height))
            valid_pixels = pixels_array[valid_mask]
            
            # Set all pixels for this segmentation at once
            for x, y in valid_pixels:
                image.setPixelColor(int(x), int(y), QColor(*segmentation.color))
        
        # Convert back to pixmap
        self.mask_pixmap = QPixmap.fromImage(image)
        
        # Invalidate cache when mask changes
        self.invalidate_scaled_mask_cache()
        
        print(f"Redrew segmentation {segmentation.id} with {len(segmentation.pixels)} pixels")
    
    def _clear_image_to_color(self, image, color):
        """Clear image to a specific color using NumPy for efficiency"""
        width, height = image.width(), image.height()
        
        # Create a QImage with the desired color
        for y in range(height):
            for x in range(width):
                image.setPixelColor(x, y, color)
    
    def set_mask_opacity(self, opacity):
        """Set the opacity for mask overlay (0.0 to 1.0)"""
        self.mask_opacity = max(0.0, min(1.0, opacity))
        self.display_image()
    
    def invalidate_scaled_mask_cache(self):
        """Invalidate the scaled mask cache when mask changes"""
        self.scaled_mask_cache = None
        self.scaled_mask_cache_zoom = None
        self.scaled_mask_cache_size = None
    
    def invalidate_scaled_selection_cache(self):
        """Invalidate the scaled selection cache when selections change"""
        self.scaled_selection_cache = None
        self.scaled_selection_cache_zoom = None
        self.scaled_selection_cache_size = None
    
    def get_scaled_mask(self, target_size):
        """Get the scaled mask, using cache if available"""
        # Check if cache is valid
        if (self.scaled_mask_cache is not None and 
            self.scaled_mask_cache_zoom == self.zoom_factor and
            self.scaled_mask_cache_size == target_size):
            return self.scaled_mask_cache
        
        # Cache miss - create new scaled mask
        if self.mask_pixmap and not self.mask_pixmap.isNull():
            scaled_mask = self.mask_pixmap.scaled(
                target_size,
                Qt.KeepAspectRatio,
                Qt.FastTransformation
            )
            
            # Update cache
            self.scaled_mask_cache = scaled_mask
            self.scaled_mask_cache_zoom = self.zoom_factor
            self.scaled_mask_cache_size = target_size
            
            return scaled_mask
        
        return None
    
    def get_scaled_selection_overlay(self, target_size):
        """Get a scaled version of the selection overlay with caching"""
        if self.selection_overlay is None:
            return None
        
        # Check if we can use the cached version
        if (self.scaled_selection_cache is not None and 
            self.scaled_selection_cache_zoom == self.zoom_factor and
            self.scaled_selection_cache_size == target_size):
            return self.scaled_selection_cache
        
        # Create a new scaled selection overlay
        height, width = self.selection_overlay.shape
        scaled_overlay = QPixmap(width, height)
        scaled_overlay.fill(Qt.transparent)
        
        # Create a painter to draw the selection overlay
        painter = QPainter(scaled_overlay)
        painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.SolidLine))  # Red pen, 2px width for highlighting
        painter.setBrush(QBrush(QColor(255, 0, 0)))  # Red brush
        
        # Draw white rectangles for each selected pixel
        for y in range(height):
            for x in range(width):
                if self.selection_overlay[y, x] > 0:  # If this is a selected pixel
                    painter.drawRect(x, y, 1, 1)
        
        painter.end()
        
        # Scale the overlay to the target size
        scaled_overlay = scaled_overlay.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.FastTransformation
        )
        
        # Cache the result
        self.scaled_selection_cache = scaled_overlay
        self.scaled_selection_cache_zoom = self.zoom_factor
        self.scaled_selection_cache_size = target_size
        
        return scaled_overlay
    

    
    def set_mode(self, mode):
        """Set the current mode (drawing, drag, paint, select, or remove)"""
        # Clear selection when switching to drawing or paint modes, but keep for drag mode
        if self.mode == "select" and mode in ["drawing", "paint"]:
            self.clear_selection()
        
        self.mode = mode
        if mode == "drawing":
            self.setCursor(Qt.CrossCursor)
        elif mode == "drag":
            self.setCursor(Qt.OpenHandCursor)
        elif mode == "paint":
            self.setCursor(Qt.PointingHandCursor)
        elif mode == "select":
            self.setCursor(Qt.PointingHandCursor)
            # Update selection mode indicator
            self.update_selection_mode_indicator(False)
        elif mode == "remove":
            self.setCursor(Qt.CrossCursor)  # Cross cursor for remove mode (like drawing)
        else:
            self.setCursor(Qt.ArrowCursor)
            self.drawing = False
            self.last_pos = None
    
    def display_image(self):
        """Display the image with optional mask overlay"""
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return
        
        # Calculate the target size based on zoom factor (integer scaling)
        original_size = self.original_pixmap.size()
        target_size = original_size * self.zoom_factor
        
        # Create a scaled version of the original image (no smoothing)
        scaled_original = self.original_pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.FastTransformation
        )
        
        # Create the display pixmap from the scaled original
        display_pixmap = QPixmap(scaled_original)
        

        

        
        # If mask is loaded, draw it with the current opacity
        scaled_mask = self.get_scaled_mask(target_size)
        if scaled_mask:
            # Create a painter to draw on the image
            painter = QPainter(display_pixmap)
            
            # Draw the mask with white color for visibility
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.setOpacity(self.mask_opacity)  # Use opacity from slider
            painter.drawPixmap(0, 0, scaled_mask)
            painter.end()
        
        # If selection overlay exists, draw selected borders in red
        scaled_selection = self.get_scaled_selection_overlay(target_size)
        if scaled_selection:
            # Create a painter to draw the selection overlay
            painter = QPainter(display_pixmap)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.drawPixmap(0, 0, scaled_selection)
            painter.end()
        
        # Create a viewport of the zoomed image
        widget_size = self.size()
        viewport_width = min(widget_size.width(), target_size.width())
        viewport_height = min(widget_size.height(), target_size.height())
        
        # Clamp viewport coordinates
        max_viewport_x = max(0, target_size.width() - viewport_width)
        max_viewport_y = max(0, target_size.height() - viewport_height)
        self.viewport_x = max(0, min(self.viewport_x, max_viewport_x))
        self.viewport_y = max(0, min(self.viewport_y, max_viewport_y))
        
        # Extract the viewport from the zoomed image
        viewport_pixmap = display_pixmap.copy(
            int(self.viewport_x), int(self.viewport_y), viewport_width, viewport_height
        )
        
        # Scale the viewport to fit the widget (this is just for display)
        final_pixmap = viewport_pixmap.scaled(
            widget_size,
            Qt.KeepAspectRatio,
            Qt.FastTransformation
        )
        self.setPixmap(final_pixmap)
    
    def resizeEvent(self, event):
        """Handle canvas resize events to maintain image scaling"""
        super().resizeEvent(event)
        self.display_image()
    
    def mousePressEvent(self, event):
        """Handle mouse press events for drawing, dragging, and painting"""
        if event.button() == Qt.LeftButton:
            if self.mode == "drawing" and self.mask_pixmap:
                self.drawing = True
                self.last_pos = event.pos()
                # Start a new drawing
                self.current_drawing = Drawing(self.next_draw_id, [], self.pen_thickness)
                self.next_draw_id += 1
                self.draw_at_position(event.pos())
            elif self.mode == "drag":
                self.drawing = True
                self.last_pos = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
            elif self.mode == "paint" and self.mask_pixmap:
                self.flood_fill_at_position(event.pos())
            elif self.mode == "select" and self.pixel_labels is not None:
                # Check if shift key is pressed
                shift_pressed = event.modifiers() & Qt.ShiftModifier
                self.select_region_at_position(event.pos(), shift_pressed)
            elif self.mode == "remove" and self.mask_pixmap:
                # Remove mode: Click on segmentation to remove it entirely, or draw black lines to erase borders
                # First check if clicking on a segmentation
                img_pos = self.widget_to_image_coords(event.pos())
                if img_pos is not None:
                    segmentation = self.get_segmentation_at_pixel(img_pos.x(), img_pos.y())
                    if segmentation is not None:
                        # Remove the entire segmentation
                        self.remove_segmentation(segmentation)
                        # Update the display
                        self.display_image()
                        # Update the right panel
                        self.update_segmentation_display()
                        # Update pixel labels array
                        self.update_pixel_labels()
                        # Mark that changes have been made
                        self.changes_made = True
                        print(f"Removed segmentation {segmentation.id} with {segmentation.get_pixel_count()} pixels")
                        # Set drawing state for potential drag removal of more segmentations
                        self.drawing = True
                        self.last_pos = event.pos()
                        return
                
                # If not clicking on a segmentation, fall back to drawing black lines
                self.drawing = True
                self.last_pos = event.pos()
                # Start a new drawing in black
                self.current_drawing = Drawing(self.next_draw_id, [], self.pen_thickness)
                self.next_draw_id += 1
                self.draw_at_position(event.pos())
        elif event.button() == Qt.RightButton:
            # Right-click drag for panning (works in any mode)
            self.drawing = True
            self.last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for drawing and dragging"""
        if self.drawing and self.last_pos:
            if self.mode == "drawing" and self.mask_pixmap:
                self.draw_line(self.last_pos, event.pos())
                self.last_pos = event.pos()
            elif self.mode == "remove" and self.mask_pixmap:
                # In remove mode, we can either draw black lines (if started on empty space) or remove segmentations (if started on a segmentation)
                # Check if we're currently drawing (started on empty space)
                if self.current_drawing is not None:
                    # We're drawing black lines
                    self.draw_line(self.last_pos, event.pos())
                    self.last_pos = event.pos()
                else:
                    # We're removing segmentations - check if we moved to a new segmentation
                    img_pos = self.widget_to_image_coords(event.pos())
                    if img_pos is not None:
                        segmentation = self.get_segmentation_at_pixel(img_pos.x(), img_pos.y())
                        if segmentation is not None:
                            # Remove the segmentation
                            self.remove_segmentation(segmentation)
                            # Update the display
                            self.display_image()
                            # Update the right panel
                            self.update_segmentation_display()
                            # Update pixel labels array
                            self.update_pixel_labels()
                            # Mark that changes have been made
                            self.changes_made = True
                            print(f"Removed segmentation {segmentation.id} with {segmentation.get_pixel_count()} pixels")
                            self.last_pos = event.pos()
            elif self.mode == "drag":
                # Handle drag panning (left-click drag mode or right-click drag)
                # Lock interface during scroll operation
                self.lock_interface("Movendo imagem...")
                
                try:
                    # Calculate drag delta
                    delta_x = event.pos().x() - self.last_pos.x()
                    delta_y = event.pos().y() - self.last_pos.y()
                    
                    # Update viewport position (inverse direction for natural panning)
                    self.viewport_x -= delta_x
                    self.viewport_y -= delta_y
                    
                    # Clamp viewport to image bounds
                    original_size = self.original_pixmap.size()
                    zoomed_size = original_size * self.zoom_factor
                    widget_size = self.size()
                    
                    max_viewport_x = max(0, zoomed_size.width() - widget_size.width())
                    max_viewport_y = max(0, zoomed_size.height() - widget_size.height())
                    
                    self.viewport_x = int(max(0, min(self.viewport_x, max_viewport_x)))
                    self.viewport_y = int(max(0, min(self.viewport_y, max_viewport_y)))
                    
                    # Update display
                    self.display_image()
                    
                    self.last_pos = event.pos()
                finally:
                    self.unlock_interface()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events for drawing and dragging"""
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.last_pos = None
            if self.mode == "drag":
                self.setCursor(Qt.OpenHandCursor)
            elif (self.mode == "drawing" or self.mode == "remove") and self.current_drawing is not None:
                # Finalize the current drawing
                if self.current_drawing.get_pixel_count() > 0:
                    self.drawings[self.current_drawing.id] = self.current_drawing
                    # Push to undo stack
                    self.push_to_undo_stack('drawing', self.current_drawing)
                self.current_drawing = None
        elif event.button() == Qt.RightButton:
            # Right-click drag release
            self.drawing = False
            self.last_pos = None
            # Restore cursor based on current mode
            if self.mode == "drawing" or self.mode == "remove":
                self.setCursor(Qt.CrossCursor)
            elif self.mode == "drag":
                self.setCursor(Qt.OpenHandCursor)
            elif self.mode == "paint":
                self.setCursor(Qt.PointingHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
    
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming"""
        if self.original_pixmap and not self.original_pixmap.isNull():
            # Lock interface during zoom operation
            self.lock_interface("Aplicando zoom...")
            
            try:
                # Get zoom factor from wheel delta (integer steps)
                delta = event.angleDelta().y()
                zoom_change = 1 if delta > 0 else -1
                
                # Calculate new zoom factor
                new_zoom = self.zoom_factor + zoom_change
                
                # Clamp zoom factor to min/max values
                new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
                
                # Only update if zoom actually changed
                if new_zoom != self.zoom_factor:
                    # Store old zoom for viewport adjustment
                    old_zoom = self.zoom_factor
                    self.zoom_factor = new_zoom
                    
                    # Get mouse position relative to the widget
                    mouse_pos = event.pos()
                    
                    # Calculate the mouse position in absolute image coordinates
                    mouse_in_zoomed_x = self.viewport_x + mouse_pos.x()
                    mouse_in_zoomed_y = self.viewport_y + mouse_pos.y()
                    
                    # Convert to absolute image coordinates (not zoomed)
                    mouse_abs_x = mouse_in_zoomed_x // old_zoom
                    mouse_abs_y = mouse_in_zoomed_y // old_zoom
                    
                    # Get canvas dimensions
                    widget_size = self.size()
                    canvas_width = widget_size.width()
                    canvas_height = widget_size.height()
                    
                    # Calculate new viewport using the specified formula:
                    # new_left = new_zoom * ((old_zoom * x) + left) - canvas_width//2
                    # new_top = new_zoom * ((old_zoom * y) + top) - canvas_height//2
                    new_viewport_x = new_zoom * (mouse_abs_x + self.viewport_x)/old_zoom - mouse_abs_x
                    new_viewport_y = new_zoom * (mouse_abs_y + self.viewport_y)/old_zoom - mouse_abs_y
                    
                    # Debug print values
                    print(f"=== ZOOM DEBUG ===")
                    print(f"Old zoom: {old_zoom}, New zoom: {new_zoom}")
                    print(f"Mouse widget coords: ({mouse_pos.x()}, {mouse_pos.y()})")
                    print(f"Mouse zoomed coords: ({mouse_in_zoomed_x}, {mouse_in_zoomed_y})")
                    print(f"Mouse abs coords: ({mouse_abs_x}, {mouse_abs_y})")
                    print(f"Current viewport: ({self.viewport_x}, {self.viewport_y})")
                    print(f"New viewport: ({new_viewport_x}, {new_viewport_y})")
                    print(f"Canvas size: ({canvas_width}, {canvas_height})")
                    print(f"==================")
                    
                    # Clamp viewport to image bounds
                    original_size = self.original_pixmap.size()
                    zoomed_size = original_size * self.zoom_factor
                    
                    max_viewport_x = max(0, zoomed_size.width() - widget_size.width())
                    max_viewport_y = max(0, zoomed_size.height() - widget_size.height())
                    
                    self.viewport_x = int(max(0, min(new_viewport_x, max_viewport_x)))
                    self.viewport_y = int(max(0, min(new_viewport_y, max_viewport_y)))
                    
                    self.display_image()
            finally:
                self.unlock_interface()
    
    def draw_at_position(self, pos):
        """Draw a point at the given position with pen thickness"""
        if not self.mask_pixmap or not self.original_pixmap:
            return
        
        # Convert widget coordinates to image coordinates
        img_pos = self.widget_to_image_coords(pos)
        if img_pos is None:
            return
        
        # Create a painter for the mask
        painter = QPainter(self.mask_pixmap)
        # Draw in black for remove mode, white for drawing mode
        color = QColor(0, 0, 0) if self.mode == "remove" else QColor(255, 255, 255)
        painter.setPen(QPen(color, self.pen_thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPoint(img_pos.x(), img_pos.y())
        painter.end()
        
        # Invalidate cache when mask changes
        self.invalidate_scaled_mask_cache()
        
        # Update the display
        self.display_image()
    
    def draw_line(self, start_pos, end_pos):
        """Draw a line between two positions with pen thickness"""
        if not self.mask_pixmap or not self.original_pixmap:
            return
        
        # Convert widget coordinates to image coordinates
        start_img = self.widget_to_image_coords(start_pos)
        end_img = self.widget_to_image_coords(end_pos)
        
        if start_img is None or end_img is None:
            return
        
        # Create a painter for the mask
        painter = QPainter(self.mask_pixmap)
        # Draw in black for remove mode, white for drawing mode
        color = QColor(0, 0, 0) if self.mode == "remove" else QColor(255, 255, 255)
        painter.setPen(QPen(color, self.pen_thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(start_img, end_img)
        painter.end()
        
        # Add pixels to current drawing if it exists
        if self.current_drawing is not None:
            # Get pixels along the line (simplified - just add start and end points)
            # In a more sophisticated implementation, you'd get all pixels along the line
            self.current_drawing.add_pixels([(start_img.x(), start_img.y()), (end_img.x(), end_img.y())])
        
        # Invalidate cache when mask changes
        self.invalidate_scaled_mask_cache()
        
        # Update the display
        self.display_image()
    
    def widget_to_image_coords(self, widget_pos):
        """Convert widget coordinates to image coordinates"""
        if not self.original_pixmap or not self.pixmap():
            return None
        
        # Get the current displayed pixmap
        displayed_pixmap = self.pixmap()
        if not displayed_pixmap:
            return None
        
        # Calculate the offset to center the image
        widget_size = self.size()
        pixmap_size = displayed_pixmap.size()
        
        offset_x = (widget_size.width() - pixmap_size.width()) // 2
        offset_y = (widget_size.height() - pixmap_size.height()) // 2
        
        # Adjust for the offset
        adjusted_x = widget_pos.x() - offset_x
        adjusted_y = widget_pos.y() - offset_y
        
        # Check if the point is within the displayed image
        if (adjusted_x < 0 or adjusted_x >= pixmap_size.width() or
            adjusted_y < 0 or adjusted_y >= pixmap_size.height()):
            return None
        
        # Calculate the actual zoomed image size
        original_size = self.original_pixmap.size()
        zoomed_size = original_size * self.zoom_factor
        
        # Convert widget coordinates to zoomed image coordinates (accounting for viewport)
        zoomed_x = self.viewport_x + adjusted_x
        zoomed_y = self.viewport_y + adjusted_y
        
        # Convert zoomed coordinates back to original image coordinates
        img_x = int(zoomed_x / self.zoom_factor)
        img_y = int(zoomed_y / self.zoom_factor)
        
        return QPoint(img_x, img_y)
    
    def flood_fill_at_position(self, pos):
        """Flood fill at the given position if the pixel is black"""
        if not self.mask_pixmap:
            return
        
        # Convert widget coordinates to image coordinates
        img_pos = self.widget_to_image_coords(pos)
        if img_pos is None:
            return
        
        # Check if the pixel at the clicked position is black
        if self.is_pixel_black(img_pos.x(), img_pos.y()):
            # Lock the interface during operation
            self.lock_interface("Processando flood fill...")
            
            try:
                # Generate a random color
                random_color = self.generate_random_color()
                # Perform flood fill and get filled pixels
                filled_pixels = self.flood_fill(img_pos.x(), img_pos.y(), random_color)
                
                if filled_pixels:
                    # Create a new Segmentation object
                    color_tuple = (random_color.red(), random_color.green(), random_color.blue())
                    segmentation = Segmentation(self.next_seg_id, color_tuple, filled_pixels)
                    self.segmentations[self.next_seg_id] = segmentation
                    self.next_seg_id += 1
                    
                    # Compute borders for the new segmentation
                    self.compute_segmentation_borders(segmentation)
                    
                    # Push to undo stack
                    self.push_to_undo_stack('segmentation', segmentation)
                    
                    # Update the display
                    self.display_image()
                    
                    # Update the right panel to show segmentations
                    self.update_segmentation_display()
                    
                    # Update pixel labels array
                    self.update_pixel_labels()
            finally:
                # Unlock the interface
                self.unlock_interface()
    
    def is_pixel_black(self, x, y):
        """Check if a pixel is black (or very dark)"""
        if not self.mask_pixmap:
            return False
        
        # Ensure coordinates are within bounds
        if x < 0 or x >= self.mask_pixmap.width() or y < 0 or y >= self.mask_pixmap.height():
            return False
        
        # Get the pixel color
        color = self.mask_pixmap.toImage().pixelColor(x, y)
        # Check if the pixel is black (or very dark)
        return color.red() < 10 and color.green() < 10 and color.blue() < 10
    
    def generate_random_color(self):
        """Generate a random color (avoiding very dark colors)"""
        # Generate colors with minimum brightness to avoid very dark colors
        r = random.randint(32, 240)
        g = random.randint(32, 240)
        b = random.randint(32, 240)
        return QColor(r, g, b)
    
    def flood_fill(self, start_x, start_y, fill_color):
        """Perform flood fill algorithm and return filled pixels using NumPy for efficiency"""
        if not self.mask_pixmap:
            return []
        
        # Convert pixmap to image for pixel manipulation
        image = self.mask_pixmap.toImage()
        width = image.width()
        height = image.height()
        
        # Get the target color (black)
        target_color = image.pixelColor(start_x, start_y)
        
        # Convert QImage to NumPy array for faster processing
        # Create a boolean mask where target color pixels are True
        target_mask = np.zeros((height, width), dtype=bool)
        
        # Fill the target mask efficiently
        for y in range(height):
            for x in range(width):
                if image.pixelColor(x, y) == target_color:
                    target_mask[y, x] = True
        
        # Use NumPy-based flood fill
        filled_pixels = self._numpy_flood_fill(start_x, start_y, target_mask)
        
        # Apply the fill color to all filled pixels efficiently
        if filled_pixels:
            # Convert filled pixels to NumPy array for batch operations
            filled_array = np.array(filled_pixels)
            y_coords = filled_array[:, 1]
            x_coords = filled_array[:, 0]
            
            # Set all filled pixels at once
            for x, y in filled_pixels:
                image.setPixelColor(x, y, fill_color)
        
        # Convert back to pixmap
        self.mask_pixmap = QPixmap.fromImage(image)
        
        # Invalidate cache when mask changes
        self.invalidate_scaled_mask_cache()
        
        return filled_pixels
    
    def _numpy_flood_fill(self, start_x, start_y, target_mask):
        """NumPy-based flood fill implementation"""
        height, width = target_mask.shape
        
        # Initialize visited mask
        visited = np.zeros_like(target_mask, dtype=bool)
        
        # Use stack for flood fill
        stack = [(start_x, start_y)]
        filled_pixels = []
        
        while stack:
            x, y = stack.pop()
            
            # Check bounds
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            
            # Check if already visited or not target color
            if visited[y, x] or not target_mask[y, x]:
                continue
            
            # Mark as visited and add to filled pixels
            visited[y, x] = True
            filled_pixels.append((x, y))
            
            # Add neighbors to stack
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            for nx, ny in neighbors:
                if (0 <= nx < width and 0 <= ny < height and 
                    not visited[ny, nx] and target_mask[ny, nx]):
                    stack.append((nx, ny))
        
        return filled_pixels
    

    

    
    def update_segmentation_display(self):
        """Update the right panel to display segmentations"""
        # Find the main window and update right panel
        widget = self
        while widget.parent() is not None:
            widget = widget.parent()
            if hasattr(widget, 'right_panel'):
                widget.right_panel.update_segmentations(list(self.segmentations.values()))
                break
    
    def push_to_undo_stack(self, operation_type, data):
        """Push an operation to the undo stack"""
        self.undo_stack.append({
            'type': operation_type,  # 'drawing' or 'segmentation'
            'data': data
        })
        # Clear redo stack when new operation is added
        self.redo_stack.clear()
        print(f"Pushed {operation_type} to undo stack. Stack size: {len(self.undo_stack)}")
        self.update_undo_redo_buttons()
        
        # Mark that changes have been made
        self.changes_made = True
    
    def update_undo_redo_buttons(self):
        """Update the undo and redo button states"""
        # Find the main window and update buttons
        widget = self
        while widget.parent() is not None:
            widget = widget.parent()
            if hasattr(widget, 'undo_button') and hasattr(widget, 'redo_button'):
                undo_enabled = len(self.undo_stack) > 0
                redo_enabled = len(self.redo_stack) > 0
                widget.undo_button.setEnabled(undo_enabled)
                widget.redo_button.setEnabled(redo_enabled)
                print(f"Updated buttons - Undo: {undo_enabled}, Redo: {redo_enabled}")
                break
    
    def undo_last_operation(self):
        """Undo the last operation"""
        if not self.undo_stack:
            return
        
        # Get the last operation
        last_operation = self.undo_stack.pop()
        
        if last_operation['type'] == 'segmentation':
            # Remove the last segmentation
            if self.segmentations:
                # Get the last segmentation by ID
                last_seg_id = max(self.segmentations.keys())
                removed_seg = self.segmentations.pop(last_seg_id)
                # Set all pixels in the segmentation to black
                self.set_segmentation_pixels_to_black(removed_seg.pixels)
                # Move to redo stack
                self.redo_stack.append(last_operation)
                self.update_segmentation_display()
        
        elif last_operation['type'] == 'drawing':
            # Remove the last drawing
            if self.drawings:
                # Get the last drawing by ID
                last_draw_id = max(self.drawings.keys())
                removed_draw = self.drawings.pop(last_draw_id)
                # Draw black lines between pixels to properly undo the drawing
                self.draw_black_lines_between_pixels(removed_draw.pixels, removed_draw.thickness)
                # Move to redo stack
                self.redo_stack.append(last_operation)
        
        elif last_operation['type'] == 'merge':
            # Undo merge operation: restore old segmentations and remove merged one
            data = last_operation['data']
            old_segmentations = data['old_segmentations']
            new_segmentation = data['new_segmentation']
            
            # Remove the merged segmentation
            if new_segmentation.id in self.segmentations:
                del self.segmentations[new_segmentation.id]
            
            # Restore old segmentations
            for segmentation in old_segmentations:
                self.segmentations[segmentation.id] = segmentation
            
            # Move to redo stack
            self.redo_stack.append(last_operation)
            self.update_segmentation_display()
        
        # Update the display
        self.display_image()
        
        # Update pixel labels array
        self.update_pixel_labels()
        
        self.update_undo_redo_buttons()
    
    def redo_last_operation(self):
        """Redo the last undone operation"""
        if not self.redo_stack:
            return
        
        # Get the last redo operation
        last_operation = self.redo_stack.pop()
        
        if last_operation['type'] == 'segmentation':
            # Restore the segmentation
            segmentation = last_operation['data']
            self.segmentations[segmentation.id] = segmentation
            # Redraw the segmentation pixels
            self.redraw_segmentation_pixels(segmentation)
            # Move back to undo stack
            self.undo_stack.append(last_operation)
            self.update_segmentation_display()
        
        elif last_operation['type'] == 'drawing':
            # Restore the drawing
            drawing = last_operation['data']
            self.drawings[drawing.id] = drawing
            # Redraw the drawing pixels
            self.redraw_drawing_pixels(drawing)
            # Move back to undo stack
            self.undo_stack.append(last_operation)
        
        elif last_operation['type'] == 'merge':
            # Redo merge operation: remove old segmentations and restore merged one
            data = last_operation['data']
            old_segmentations = data['old_segmentations']
            new_segmentation = data['new_segmentation']
            
            # Remove old segmentations
            for segmentation in old_segmentations:
                if segmentation.id in self.segmentations:
                    del self.segmentations[segmentation.id]
            
            # Restore the merged segmentation
            self.segmentations[new_segmentation.id] = new_segmentation
            
            # Move back to undo stack
            self.undo_stack.append(last_operation)
            self.update_segmentation_display()
        
        # Update the display
        self.display_image()
        
        # Update pixel labels array
        self.update_pixel_labels()
        
        self.update_undo_redo_buttons()
    
    def draw_black_lines_between_pixels(self, pixels, thickness=1):
        """Draw black lines between consecutive pixels to properly undo drawings with thickness"""
        if not self.mask_pixmap or len(pixels) < 2:
            return
        
        # Convert pixmap to image for pixel manipulation
        image = self.mask_pixmap.toImage()
        
        # Create a painter to draw lines
        painter = QPainter(image)
        painter.setPen(QPen(QColor(0, 0, 0), thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        
        # Draw lines between consecutive pixels
        for i in range(len(pixels) - 1):
            x1, y1 = pixels[i]
            x2, y2 = pixels[i + 1]
            if (0 <= x1 < image.width() and 0 <= y1 < image.height() and
                0 <= x2 < image.width() and 0 <= y2 < image.height()):
                painter.drawLine(x1, y1, x2, y2)
        
        painter.end()
        
        # Convert back to pixmap
        self.mask_pixmap = QPixmap.fromImage(image)
        
        # Invalidate cache when mask changes
        self.invalidate_scaled_mask_cache()
    
    def set_segmentation_pixels_to_black(self, pixels):
        """Set specified pixels to black for segmentation undo using NumPy for efficiency"""
        if not self.mask_pixmap or not pixels:
            return
        
        # Convert pixmap to image for pixel manipulation
        image = self.mask_pixmap.toImage()
        width, height = image.width(), image.height()
        
        # Convert pixels to NumPy array for batch operations
        pixels_array = np.array(pixels)
        
        # Filter valid pixels (within bounds)
        valid_mask = ((pixels_array[:, 0] >= 0) & (pixels_array[:, 0] < width) & 
                     (pixels_array[:, 1] >= 0) & (pixels_array[:, 1] < height))
        valid_pixels = pixels_array[valid_mask]
        
        # Set all valid pixels to black at once
        for x, y in valid_pixels:
            image.setPixelColor(int(x), int(y), QColor(0, 0, 0))
        
        # Convert back to pixmap
        self.mask_pixmap = QPixmap.fromImage(image)
        
        # Invalidate cache when mask changes
        self.invalidate_scaled_mask_cache()
    
    def redraw_segmentation_pixels(self, segmentation):
        """Redraw segmentation pixels with their original color using NumPy for efficiency"""
        if not self.mask_pixmap or not segmentation.pixels:
            return
        
        # Convert pixmap to image for pixel manipulation
        image = self.mask_pixmap.toImage()
        width, height = image.width(), image.height()
        
        # Convert segmentation pixels to NumPy array for batch operations
        pixels_array = np.array(segmentation.pixels)
        
        # Filter valid pixels (within bounds)
        valid_mask = ((pixels_array[:, 0] >= 0) & (pixels_array[:, 0] < width) & 
                     (pixels_array[:, 1] >= 0) & (pixels_array[:, 1] < height))
        valid_pixels = pixels_array[valid_mask]
        
        # Set all valid pixels with original color at once
        for x, y in valid_pixels:
            image.setPixelColor(int(x), int(y), QColor(*segmentation.color))
        
        # Convert back to pixmap
        self.mask_pixmap = QPixmap.fromImage(image)
        
        # Invalidate cache when mask changes
        self.invalidate_scaled_mask_cache()
    
    def redraw_drawing_pixels(self, drawing):
        """Redraw drawing pixels with white color by drawing lines between pixels using thickness"""
        if not self.mask_pixmap or len(drawing.pixels) < 2:
            return
        
        # Convert pixmap to image for pixel manipulation
        image = self.mask_pixmap.toImage()
        
        # Create a painter to draw lines
        painter = QPainter(image)
        painter.setPen(QPen(QColor(255, 255, 255), drawing.thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        
        # Draw lines between consecutive pixels
        for i in range(len(drawing.pixels) - 1):
            x1, y1 = drawing.pixels[i]
            x2, y2 = drawing.pixels[i + 1]
            if (0 <= x1 < image.width() and 0 <= y1 < image.height() and
                0 <= x2 < image.width() and 0 <= y2 < image.height()):
                painter.drawLine(x1, y1, x2, y2)
        
        painter.end()
        
        # Convert back to pixmap
        self.mask_pixmap = QPixmap.fromImage(image)
    
        # Invalidate cache when mask changes
        self.invalidate_scaled_mask_cache()
    
    def has_unsaved_changes(self):
        """Check if there are unsaved changes"""
        return self.changes_made
    
    def mark_changes_saved(self):
        """Mark changes as saved"""
        self.changes_made = False
    
    def initialize_pixel_labels(self):
        """Initialize the pixel labels array based on image size"""
        if self.original_pixmap:
            width = self.original_pixmap.width()
            height = self.original_pixmap.height()
            self.pixel_labels = np.zeros((height, width), dtype=np.int32)
            print(f"Initialized pixel labels array: {width}x{height}")
        else:
            self.pixel_labels = None
    
    def update_pixel_labels(self):
        """Update the pixel labels array based on current segmentations"""
        if self.pixel_labels is None:
            self.initialize_pixel_labels()
        
        if self.pixel_labels is None:
            return
        
        # Clear the array
        self.pixel_labels.fill(0)
        
        # Fill in segmentation labels
        for seg_id, segmentation in self.segmentations.items():
            for x, y in segmentation.pixels:
                if 0 <= y < self.pixel_labels.shape[0] and 0 <= x < self.pixel_labels.shape[1]:
                    self.pixel_labels[y, x] = seg_id
        
        print(f"Updated pixel labels array with {len(self.segmentations)} segmentations")
    
    def get_pixel_label(self, x, y):
        """Get the segmentation label for a pixel at (x, y)"""
        if self.pixel_labels is None or x < 0 or y < 0:
            return 0
        
        if y >= self.pixel_labels.shape[0] or x >= self.pixel_labels.shape[1]:
            return 0
        
        return self.pixel_labels[y, x]
    
    def get_segmentation_at_pixel(self, x, y):
        """Get the segmentation object at pixel (x, y), or None if no segmentation"""
        label = self.get_pixel_label(x, y)
        if label == 0:
            return None
        return self.segmentations.get(label)
    
    def get_segmentation_statistics(self):
        """Get statistics about the pixel labels array"""
        if self.pixel_labels is None:
            return {
                'total_pixels': 0,
                'labeled_pixels': 0,
                'unlabeled_pixels': 0,
                'unique_labels': 0,
                'label_counts': {}
            }
        
        total_pixels = self.pixel_labels.size
        labeled_pixels = np.sum(self.pixel_labels > 0)
        unlabeled_pixels = total_pixels - labeled_pixels
        unique_labels = len(np.unique(self.pixel_labels[self.pixel_labels > 0]))
        
        # Count pixels per label
        label_counts = {}
        for label in np.unique(self.pixel_labels):
            if label > 0:  # Skip background (0)
                count = np.sum(self.pixel_labels == label)
                label_counts[label] = int(count)
        
        return {
            'total_pixels': int(total_pixels),
            'labeled_pixels': int(labeled_pixels),
            'unlabeled_pixels': int(unlabeled_pixels),
            'unique_labels': int(unique_labels),
            'label_counts': label_counts
        }
    
    def compute_segmentation_borders(self, segmentation):
        """Compute border pixels for a segmentation using NumPy for efficiency"""
        if not segmentation.pixels:
            segmentation.border_pixels = []
            return
        
        # Only show progress for large segmentations (more than 5000 pixels)
        progress = None
        if len(segmentation.pixels) > 5000:
            progress = self.show_progress_dialog("Calculando Bordas", f"Processando {len(segmentation.pixels)} pixels...", len(segmentation.pixels))
        
        try:
            # Get the current segmentation ID
            seg_id = segmentation.id
            
            # Convert segmentation pixels to NumPy array for efficient operations
            pixels_array = np.array(segmentation.pixels)
            
            # Create a set of segmentation pixels for fast lookup
            # Convert any lists to tuples to make them hashable
            segmentation_set = set(tuple(pixel) if isinstance(pixel, list) else pixel for pixel in segmentation.pixels)
            
            # Find border pixels using NumPy operations
            border_pixels = []
            processed_pixels = 0
            
            # Define neighbor offsets
            neighbors = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
            
            for i, pixel in enumerate(segmentation.pixels):
                # Ensure pixel is a tuple
                if isinstance(pixel, list):
                    x, y = tuple(pixel)
                else:
                    x, y = pixel
                
                # Calculate neighbor coordinates using NumPy
                neighbor_coords = np.array([x, y]) + neighbors
                
                # Check if any neighbor is outside the segmentation
                is_border = False
                for nx, ny in neighbor_coords:
                    # Check bounds
                    if (0 <= nx < self.original_pixmap.width() and 
                        0 <= ny < self.original_pixmap.height()):
                        # Check if neighbor is outside segmentation
                        if (nx, ny) not in segmentation_set:
                            # Check if neighbor has a different label
                            neighbor_label = self.get_pixel_label(nx, ny)
                            if neighbor_label != seg_id:
                                border_pixels.append((x, y))
                                is_border = True
                                break
                
                # Update progress only if progress dialog exists
                processed_pixels += 1
                if progress and processed_pixels % 100 == 0:  # Update every 100 pixels
                    progress.setValue(processed_pixels)
                    if progress.wasCanceled():
                        break
            
            segmentation.border_pixels = border_pixels
            print(f"Computed {len(border_pixels)} border pixels for segmentation {seg_id}")
        finally:
            if progress:
                progress.close()
    
    def remove_segmentation(self, segmentation):
        """Remove a segmentation and set its pixels to black"""
        if not segmentation or not self.mask_pixmap:
            return
        
        # Store the segmentation for potential undo (though we'll clear the queue)
        removed_data = {
            'segmentation': segmentation,
            'pixels': segmentation.pixels.copy()
        }
        
        # Set the segmentation pixels to black on the mask
        image = self.mask_pixmap.toImage()
        for x, y in segmentation.pixels:
            if 0 <= x < image.width() and 0 <= y < image.height():
                image.setPixelColor(x, y, QColor(0, 0, 0))  # Black
        
        # Convert back to pixmap
        self.mask_pixmap = QPixmap.fromImage(image)
        
        # Remove from segmentations dictionary
        seg_id = segmentation.id
        if seg_id in self.segmentations:
            del self.segmentations[seg_id]
        
        # Remove from selected segmentations if it was selected
        if seg_id in self.selected_segmentations:
            self.selected_segmentations.remove(seg_id)
        
        # Clear the undo and redo stacks
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.update_undo_redo_buttons()
        
        # Update pixel labels array
        self.update_pixel_labels()
        
        # Clear selection overlay
        self.selection_overlay = None
        self.invalidate_scaled_selection_cache()
        
        # Invalidate mask cache
        self.invalidate_scaled_mask_cache()
        
        # Mark that changes have been made
        self.changes_made = True
        
        print(f"Removed segmentation {seg_id} with {len(removed_data['pixels'])} pixels")
    
    def select_region_at_position(self, pos, shift_pressed=False):
        """Select a region at the given position"""
        # Lock interface during selection operation
        self.lock_interface("Selecionando região...")
        
        try:
            # Convert widget coordinates to image coordinates
            img_pos = self.widget_to_image_coords(pos)
            if img_pos is None:
                return
            
            # Get the segmentation at this pixel
            segmentation = self.get_segmentation_at_pixel(img_pos.x(), img_pos.y())
            
            if segmentation:
                if shift_pressed:
                    # Shift key is held - toggle selection (add/remove from current selection)
                    if segmentation.id in self.selected_segmentations:
                        # Deselect this segmentation
                        self.selected_segmentations.remove(segmentation.id)
                        print(f"Deselected segmentation {segmentation.id}")
                    else:
                        # Add this segmentation to selection
                        self.selected_segmentations.add(segmentation.id)
                        print(f"Added segmentation {segmentation.id} to selection ({len(self.selected_segmentations)} total)")
                else:
                    # Shift key not held - toggle selection for this segmentation
                    if segmentation.id in self.selected_segmentations:
                        # Already selected - deselect it
                        self.selected_segmentations.remove(segmentation.id)
                        print(f"Deselected segmentation {segmentation.id}")
                    else:
                        # Not selected - clear previous selections and select only this one
                        self.selected_segmentations.clear()
                        self.selected_segmentations.add(segmentation.id)
                        print(f"Selected only segmentation {segmentation.id} with {segmentation.get_pixel_count()} pixels")
                
                # Update selection overlay for all selected segmentations
                self.update_selection_overlay()
                
                # Update merge button state
                self.update_merge_button_state()
                
                # Update the display
                self.display_image()
                
                # Update checkbox states and styling in the right panel
                if self.right_panel:
                    self.right_panel.update_checkbox_states()
                    self.right_panel.update_segmentation_item_styling()
            else:
                # Clear all selections if clicking on empty area
                if self.selected_segmentations:
                    self.selected_segmentations.clear()
                    self.border_overlay = None
                    print("Cleared all selections")
                    self.display_image()
                    
                    # Update checkbox states and styling in the right panel
                    if self.right_panel:
                        self.right_panel.update_checkbox_states()
                        self.right_panel.update_segmentation_item_styling()
        finally:
            self.unlock_interface()
    
    def update_selection_overlay(self):
        """Update selection overlay for all selected segmentations using NumPy for efficiency"""
        if self.pixel_labels is None or not self.selected_segmentations:
            self.selection_overlay = None
            return
        
        # Calculate total border pixels to determine if we need progress
        total_border_pixels = 0
        for seg_id in self.selected_segmentations:
            if seg_id in self.segmentations:
                total_border_pixels += len(self.segmentations[seg_id].border_pixels)
        
        # Only show progress for large selections (more than 5000 border pixels)
        progress = None
        if total_border_pixels > 5000:
            progress = self.show_progress_dialog("Atualizando Seleção", "Processando bordas das seleções...", len(self.selected_segmentations))
        
        try:
            # Create a selection overlay using NumPy
            height, width = self.pixel_labels.shape
            self.selection_overlay = np.zeros((height, width), dtype=np.uint8)
            
            total_selected_pixels = 0
            processed_segmentations = 0
            
            # Color border pixels of selected segmentations white using NumPy
            for seg_id in self.selected_segmentations:
                if seg_id in self.segmentations:
                    segmentation = self.segmentations[seg_id]
                    
                    if segmentation.border_pixels:
                        # Convert border pixels to NumPy array for batch operations
                        border_array = np.array(segmentation.border_pixels)
                        
                        # Filter valid pixels (within bounds)
                        valid_mask = ((border_array[:, 0] >= 0) & (border_array[:, 0] < width) & 
                                     (border_array[:, 1] >= 0) & (border_array[:, 1] < height))
                        valid_borders = border_array[valid_mask]
                        
                        # Set all border pixels at once using NumPy indexing
                        if len(valid_borders) > 0:
                            y_coords = valid_borders[:, 1]
                            x_coords = valid_borders[:, 0]
                            self.selection_overlay[y_coords, x_coords] = 255
                    
                    total_selected_pixels += len(segmentation.border_pixels)
                    print(f"Colored {len(segmentation.border_pixels)} border pixels white for segmentation {seg_id}")
                
                # Update progress only if progress dialog exists
                processed_segmentations += 1
                if progress:
                    progress.setValue(processed_segmentations)
                    if progress.wasCanceled():
                        break
            
            print(f"Total selected pixels for {len(self.selected_segmentations)} selected segmentations: {total_selected_pixels}")
            
            # Invalidate the selection cache since selections changed
            self.invalidate_scaled_selection_cache()
        finally:
            if progress:
                progress.close()
    

    
    def clear_selection(self):
        """Clear all selections"""
        self.selected_segmentations.clear()
        self.selection_overlay = None
        self.invalidate_scaled_selection_cache()
        self.update_merge_button_state()
        self.display_image()
        
        # Update checkbox states and styling in the right panel
        if self.right_panel:
            self.right_panel.update_checkbox_states()
            self.right_panel.update_segmentation_item_styling()
    
    def get_selected_segmentations(self):
        """Get list of selected segmentation objects"""
        selected = []
        for seg_id in self.selected_segmentations:
            if seg_id in self.segmentations:
                selected.append(self.segmentations[seg_id])
        return selected
    
    def is_segmentation_selected(self, seg_id):
        """Check if a segmentation is selected"""
        return seg_id in self.selected_segmentations
    
    def merge_selected_segmentations(self):
        """Merge all selected segmentations into a single new segmentation"""
        if len(self.selected_segmentations) < 2:
            print("Need at least 2 segmentations to merge")
            return
        
        # Lock interface during merge operation
        self.lock_interface("Juntando segmentações...")
        
        try:
            # Calculate total pixels to determine if we need progress
            total_pixels = 0
            for seg_id in self.selected_segmentations:
                if seg_id in self.segmentations:
                    total_pixels += len(self.segmentations[seg_id].pixels)
            
            # Only show progress for large merges (more than 5000 pixels)
            progress = None
            if total_pixels > 5000:
                progress = self.show_progress_dialog("Juntando Segmentações", "Coletando pixels das segmentações...", 3)
            
            # Step 1: Collect all pixels and borders from selected segmentations
            all_pixels = []
            all_border_pixels = []
            old_segmentations = []
            
            for seg_id in self.selected_segmentations:
                if seg_id in self.segmentations:
                    segmentation = self.segmentations[seg_id]
                    all_pixels.extend(segmentation.pixels)
                    all_border_pixels.extend(segmentation.border_pixels)
                    old_segmentations.append(segmentation)
            
            if progress:
                progress.setValue(1)
                progress.setLabelText("Criando nova segmentação...")
            
            if not all_pixels:
                print("No pixels found in selected segmentations")
                return
            
            # Step 2: Generate a new random color for the merged segmentation
            new_color = self.generate_random_color()
            color_tuple = (new_color.red(), new_color.green(), new_color.blue())
            
            # Create new merged segmentation with existing borders (no reprocessing)
            merged_segmentation = Segmentation(self.next_seg_id, color_tuple, all_pixels)
            merged_segmentation.border_pixels = all_border_pixels
            
            # Store the old segmentations for undo
            old_segmentations_data = {
                'old_segmentations': old_segmentations,
                'new_segmentation': merged_segmentation
            }
            
            if progress:
                progress.setValue(2)
                progress.setLabelText("Removendo segmentações antigas...")
            
            # Step 3: Remove old segmentations from the dictionary
            for seg_id in self.selected_segmentations:
                if seg_id in self.segmentations:
                    del self.segmentations[seg_id]
            
            # Add the new merged segmentation
            self.segmentations[self.next_seg_id] = merged_segmentation
            self.next_seg_id += 1
            
            # Clear selections
            self.selected_segmentations.clear()
            self.selection_overlay = None
            
            # Update merge button state
            self.update_merge_button_state()
            
            # Push to undo stack
            self.push_to_undo_stack('merge', old_segmentations_data)
            
            if progress:
                progress.setValue(3)
                progress.setLabelText("Finalizando...")
            
            # Update pixel labels array
            self.update_pixel_labels()
            
            # Redraw only the merged segmentation on the mask (not all segmentations)
            self.redraw_single_segmentation(merged_segmentation)
            
            # Update the display
            self.display_image()
            
            # Update the right panel
            self.update_segmentation_display()
            
            # Update checkbox states and styling in the right panel
            if self.right_panel:
                self.right_panel.update_checkbox_states()
                self.right_panel.update_segmentation_item_styling()
            
            print(f"Merged {len(old_segmentations)} segmentations into new segmentation {merged_segmentation.id} with {len(all_pixels)} pixels")
        finally:
            if progress:
                progress.close()
            self.unlock_interface()
    
    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key_Shift:
            # Update cursor to indicate shift mode
            if self.mode == "select":
                self.setCursor(Qt.CrossCursor)  # Different cursor for shift mode
                # Update right panel selection mode indicator
                self.update_selection_mode_indicator(True)
        super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event):
        """Handle key release events"""
        if event.key() == Qt.Key_Shift:
            # Restore normal cursor
            if self.mode == "select":
                self.setCursor(Qt.PointingHandCursor)
                # Update right panel selection mode indicator
                self.update_selection_mode_indicator(False)
        super().keyReleaseEvent(event)
    
    def update_selection_mode_indicator(self, shift_pressed=False):
        """Update the selection mode indicator in the right panel"""
        # This method is kept for compatibility but no longer updates the UI
        pass
    
    def update_merge_button_state(self):
        """Update the merge button enabled state based on selections"""
        # Find the main window and update merge button
        widget = self
        while widget.parent() is not None:
            widget = widget.parent()
            if hasattr(widget, 'merge_button'):
                # Enable merge button if 2 or more segmentations are selected
                can_merge = len(self.selected_segmentations) >= 2
                widget.merge_button.setEnabled(can_merge)
                break
    
    def lock_interface(self, message="Processando..."):
        """Lock the interface to prevent user input during operations"""
        # Find the main window
        widget = self
        while widget.parent() is not None:
            widget = widget.parent()
            if hasattr(widget, 'setEnabled'):
                # Disable the main window
                widget.setEnabled(False)
                
                # Show a status message if available
                if hasattr(widget, 'statusBar'):
                    widget.statusBar().showMessage(message)
                elif hasattr(widget, 'setWindowTitle'):
                    # Store original title and show processing message
                    if not hasattr(widget, '_original_title'):
                        widget._original_title = widget.windowTitle()
                    widget.setWindowTitle(f"{widget._original_title} - {message}")
                break
    
    def unlock_interface(self):
        """Unlock the interface after operations are complete"""
        # Find the main window
        widget = self
        while widget.parent() is not None:
            widget = widget.parent()
            if hasattr(widget, 'setEnabled'):
                # Enable the main window
                widget.setEnabled(True)
                
                # Clear status message if available
                if hasattr(widget, 'statusBar'):
                    widget.statusBar().clearMessage()
                elif hasattr(widget, 'setWindowTitle') and hasattr(widget, '_original_title'):
                    # Restore original title
                    widget.setWindowTitle(widget._original_title)
                break
    
    def show_progress_dialog(self, title, message, maximum=100):
        """Show a progress dialog for longer operations"""
        # Find the main window
        widget = self
        while widget.parent() is not None:
            widget = widget.parent()
            if hasattr(widget, 'setEnabled'):
                # Create progress dialog
                progress = QProgressDialog(message, "Cancelar", 0, maximum, widget)
                progress.setWindowTitle(title)
                progress.setWindowModality(Qt.WindowModal)
                progress.setAutoClose(True)
                progress.setAutoReset(True)
                progress.setMinimumDuration(0)  # Show immediately
                return progress
        return None
    




class RightPanel(QFrame):
    """Right panel widget to display segmentations"""
    
    def __init__(self):
        super().__init__()
        self.canvas = None  # Will be set by set_canvas method
        self.setMinimumWidth(350)
        self.setMaximumWidth(450)
        self.setStyleSheet("""
            QFrame {
                border: 2px solid #cccccc;
                background-color: #f8f8f8;
                border-radius: 5px;
            }
        """)
        
        # Create layout
        self.layout = QVBoxLayout()
        
        # Add title
        title = QLabel("Rotulações")
        title.setStyleSheet("color: #333333; font-size: 16px; font-weight: bold; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title)
        
        # Add statistics label
        self.stats_label = QLabel("Rotulados: 0\nCobertura: 0.0%")
        self.stats_label.setStyleSheet("""
            QLabel {
                color: #666666;
                font-size: 12px;
                padding: 8px 10px;
                background-color: #e8e8e8;
                border: 1px solid #cccccc;
                border-radius: 3px;
                margin: 0px 10px;
            }
        """)
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setFixedHeight(80)
        self.layout.addWidget(self.stats_label)
        
        # Add scrollable area for segmentations
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
        # Create widget to hold segmentation items
        self.segmentation_widget = QWidget()
        self.segmentation_layout = QVBoxLayout(self.segmentation_widget)
        self.segmentation_layout.setAlignment(Qt.AlignTop)
        
        self.scroll_area.setWidget(self.segmentation_widget)
        self.layout.addWidget(self.scroll_area)
        
        self.setLayout(self.layout)
    
    def set_canvas(self, canvas):
        """Set the canvas reference"""
        self.canvas = canvas
    
    def set_pixel_size(self, pixel_size_mm):
        """Set the pixel size for mm² calculations"""
        self.pixel_size_mm = pixel_size_mm
    
    def update_segmentations(self, segmentations):
        """Update the display with current segmentations"""
        # Clear existing items
        for i in reversed(range(self.segmentation_layout.count())):
            self.segmentation_layout.itemAt(i).widget().setParent(None)
        
        # Clear references
        if hasattr(self, 'segmentation_checkboxes'):
            self.segmentation_checkboxes.clear()
        if hasattr(self, 'segmentation_color_labels'):
            self.segmentation_color_labels.clear()
        if hasattr(self, 'segmentation_info_labels'):
            self.segmentation_info_labels.clear()
        
        # Add segmentation items
        for seg in segmentations:
            self.add_segmentation_item(seg)
        
        # Update statistics
        self.update_statistics(segmentations)
    
    def add_segmentation_item(self, segmentation):
        """Add a single segmentation item to the display"""
        # Create item widget
        item_widget = QWidget()
        item_layout = QHBoxLayout(item_widget)
        
        # Check if this segmentation is selected
        is_selected = False
        if self.canvas and self.canvas.is_segmentation_selected(segmentation.id):
            is_selected = True
        
        # Checkbox for selection
        checkbox = QCheckBox()
        checkbox.setChecked(is_selected)
        checkbox.setStyleSheet("""
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #cccccc;
                background-color: white;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #FF5722;
                background-color: #FF5722;
                border-radius: 3px;
            }
        """)
        
        # Connect checkbox to selection function
        checkbox.toggled.connect(lambda checked, seg_id=segmentation.id: self.toggle_segmentation_selection(seg_id, checked))
        item_layout.addWidget(checkbox)
        
        # Color indicator
        color_label = QLabel()
        color_label.setFixedSize(20, 20)
        border_color = "#FF5722" if is_selected else "#333333"  # Red border if selected
        border_width = "3px" if is_selected else "1px"
        color_label.setStyleSheet(f"""
            QLabel {{
                background-color: rgb{segmentation.color};
                border: {border_width} solid {border_color};
                border-radius: 3px;
            }}
        """)
        item_layout.addWidget(color_label)
        
        # Area in mm²
        pixel_count = segmentation.get_pixel_count()
        pixel_size_mm = getattr(self, 'pixel_size_mm', 0.0)
        area_mm2 = pixel_count * (pixel_size_mm ** 2)
        
        # Display ? mm² when pixel size is 0
        if pixel_size_mm == 0.0:
            area_text = "? mm²"
        else:
            area_text = f"{area_mm2:.2f} mm²"
        
        info_label = QLabel(area_text)
        text_color = "#FF5722" if is_selected else "#333333"  # Red text if selected
        font_weight = "bold" if is_selected else "normal"
        info_label.setStyleSheet(f"color: {text_color}; font-size: 12px; font-weight: {font_weight};")
        item_layout.addWidget(info_label)
        
        # Pixel count in parentheses
        count_label = QLabel(f"({pixel_count} pixels)")
        count_label.setStyleSheet("color: #666666; font-size: 11px;")
        item_layout.addWidget(count_label)
        
        item_layout.addStretch()
        
        # Store references for later updates
        if not hasattr(self, 'segmentation_checkboxes'):
            self.segmentation_checkboxes = {}
        if not hasattr(self, 'segmentation_color_labels'):
            self.segmentation_color_labels = {}
        if not hasattr(self, 'segmentation_info_labels'):
            self.segmentation_info_labels = {}
        
        self.segmentation_checkboxes[segmentation.id] = checkbox
        self.segmentation_color_labels[segmentation.id] = color_label
        self.segmentation_info_labels[segmentation.id] = info_label
        
        self.segmentation_layout.addWidget(item_widget)
    
    def toggle_segmentation_selection(self, seg_id, checked):
        """Handle checkbox toggle for segmentation selection"""
        if self.canvas:
            if checked:
                # Add to selection
                self.canvas.selected_segmentations.add(seg_id)
            else:
                # Remove from selection
                self.canvas.selected_segmentations.discard(seg_id)
            
            # Update selection overlay and merge button state
            self.canvas.update_selection_overlay()
            self.canvas.update_merge_button_state()
            self.canvas.update_selection_mode_indicator()
            
            # Update the display
            self.canvas.display_image()
            self.update_statistics(list(self.canvas.segmentations.values()))
            
            # Update visual styling of segmentation items
            self.update_segmentation_item_styling()
    
    def update_checkbox_states(self):
        """Update all checkbox states to match current selection"""
        if hasattr(self, 'segmentation_checkboxes') and self.canvas:
            for seg_id, checkbox in self.segmentation_checkboxes.items():
                is_selected = seg_id in self.canvas.selected_segmentations
                checkbox.setChecked(is_selected)
    
    def update_segmentation_item_styling(self):
        """Update visual styling of segmentation items based on selection state"""
        if hasattr(self, 'segmentation_color_labels') and hasattr(self, 'segmentation_info_labels') and self.canvas:
            for seg_id in self.segmentation_color_labels.keys():
                is_selected = seg_id in self.canvas.selected_segmentations
                
                # Update color label styling
                if seg_id in self.segmentation_color_labels:
                    color_label = self.segmentation_color_labels[seg_id]
                    border_color = "#FF5722" if is_selected else "#333333"
                    border_width = "3px" if is_selected else "1px"
                    color_label.setStyleSheet(f"""
                        QLabel {{
                            background-color: rgb{self.canvas.segmentations[seg_id].color};
                            border: {border_width} solid {border_color};
                            border-radius: 3px;
                        }}
                    """)
                
                # Update info label styling
                if seg_id in self.segmentation_info_labels:
                    info_label = self.segmentation_info_labels[seg_id]
                    text_color = "#FF5722" if is_selected else "#333333"
                    font_weight = "bold" if is_selected else "normal"
                    info_label.setStyleSheet(f"color: {text_color}; font-size: 12px; font-weight: {font_weight};")
    
    def update_statistics(self, segmentations):
        """Update the statistics label with current segmentation data"""
        num_segmentations = len(segmentations)
        
        # Calculate total labeled area
        total_labeled_area = sum(seg.get_pixel_count() for seg in segmentations)
        
        # Calculate coverage percentage
        coverage_percentage = 0.0
        coverage_mm2 = 0.0
        if self.canvas and self.canvas.original_pixmap:
            # Get total image area
            total_image_area = self.canvas.original_pixmap.width() * self.canvas.original_pixmap.height()
            if total_image_area > 0:
                coverage_percentage = (total_labeled_area / total_image_area) * 100.0
                # Calculate mm² using pixel size
                pixel_size_mm = getattr(self, 'pixel_size_mm', 0.0)
                if pixel_size_mm == 0.0:
                    coverage_mm2 = 0.0  # Will display as ? mm²
                else:
                    coverage_mm2 = total_labeled_area * (pixel_size_mm ** 2)
        
        # Get selection information
        selection_info = ""
        selected_area_sum = 0
        selected_area_mm2 = 0
        if self.canvas:
            selected_count = len(self.canvas.selected_segmentations)
            if selected_count > 0:
                # Calculate sum of selected areas
                for seg_id in self.canvas.selected_segmentations:
                    if seg_id in self.canvas.segmentations:
                        selected_area_sum += self.canvas.segmentations[seg_id].get_pixel_count()
                # Calculate mm² for selected areas
                pixel_size_mm = getattr(self, 'pixel_size_mm', 0.0)
                if pixel_size_mm == 0.0:
                    selected_area_mm2 = 0.0  # Will display as ? mm²
                else:
                    selected_area_mm2 = selected_area_sum * (pixel_size_mm ** 2)
                selection_info = f"\nSelecionadas: {selected_count} regiões\nÁrea Selecionada: {selected_area_sum} pixels ({'? mm²' if pixel_size_mm == 0.0 else f'{selected_area_mm2:.2f} mm²'})"
        
        # Format coverage area display
        pixel_size_mm = getattr(self, 'pixel_size_mm', 0.0)
        coverage_text = "? mm²" if pixel_size_mm == 0.0 else f"{coverage_mm2:.2f} mm²"
        self.stats_label.setText(f"Rotulados: {num_segmentations}\nCobertura: {coverage_percentage:.1f}% ({coverage_text}){selection_info}")


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.last_opacity_value = 50  # Store the last non-zero opacity value
        self.init_ui()
    
    def get_base_directory(self):
        """Get the base directory that works with both development and PyInstaller"""
        # Try to get the directory of the executable or script
        if getattr(sys, 'frozen', False):
            # Running as PyInstaller bundle
            base_dir = os.path.dirname(sys.executable)
        else:
            # Running as script
            base_dir = os.path.dirname(os.path.abspath(__file__))
        
        return base_dir
    
    def update_window_title(self):
        """Update the window title to show loaded image and mask names"""
        title = "Anotador de Imagem"
        
        # Add image name if loaded
        if hasattr(self, 'current_image_path') and self.current_image_path:
            image_name = os.path.basename(self.current_image_path)
            title += f" - {image_name}"
        
        # Add mask name if loaded
        if hasattr(self, 'current_mask_path') and self.current_mask_path:
            mask_name = os.path.basename(self.current_mask_path)
            title += f" | {mask_name}"
        
        self.setWindowTitle(title)
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Anotador de Imagem")
        self.setGeometry(100, 100, 800, 600)
        
        # Create status bar
        self.statusBar = self.statusBar()
        self.statusBar.setStyleSheet("""
            QStatusBar {
                background-color: #f0f0f0;
                border-top: 1px solid #cccccc;
                padding: 2px;
            }
        """)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Top section with 2 rows
        top_layout = QVBoxLayout()
        
        # First row: Load, Save, and Mode selection
        first_row = QHBoxLayout()
        
        # Load button
        self.load_button = QPushButton("Carregar Imagem")
        self.load_button.setFixedSize(140, 40)
        self.load_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.load_button.clicked.connect(self.load_image)
        
        first_row.addWidget(self.load_button)
        
        # Add some spacing
        first_row.addSpacing(20)
        
        # Load mask button
        self.load_mask_button = QPushButton("Carregar Máscara")
        self.load_mask_button.setFixedSize(140, 40)
        self.load_mask_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:pressed {
                background-color: #EF6C00;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.load_mask_button.setEnabled(False)  # Disabled until image is loaded
        self.load_mask_button.clicked.connect(self.load_mask)
        first_row.addWidget(self.load_mask_button)
        
        # Add some spacing
        first_row.addSpacing(20)
        
        # Save mask button
        self.save_button = QPushButton("Salvar Máscara")
        self.save_button.setFixedSize(120, 40)
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.save_button.setEnabled(False)  # Disabled until image is loaded
        self.save_button.clicked.connect(self.save_mask)
        first_row.addWidget(self.save_button)
        
        # Add some spacing
        first_row.addSpacing(20)
        
        # Pen thickness slider section
        pen_label = QLabel("Espessura:")
        pen_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #333333;
                padding: 5px;
            }
        """)
        first_row.addWidget(pen_label)
        
        # Pen thickness slider
        self.pen_thickness_slider = QSlider(Qt.Horizontal)
        self.pen_thickness_slider.setMinimum(1)
        self.pen_thickness_slider.setMaximum(20)
        self.pen_thickness_slider.setValue(3)  # Default thickness
        self.pen_thickness_slider.setFixedWidth(100)
        self.pen_thickness_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #cccccc;
                height: 8px;
                background: #e0e0e0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #4CAF50;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #45a049;
            }
        """)
        self.pen_thickness_slider.valueChanged.connect(self.on_pen_thickness_changed)
        first_row.addWidget(self.pen_thickness_slider)
        
        # Pen thickness value label
        self.pen_thickness_label = QLabel("3")
        self.pen_thickness_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #333333;
                padding: 5px;
                min-width: 20px;
            }
        """)
        first_row.addWidget(self.pen_thickness_label)
        
        # Add some spacing
        first_row.addSpacing(20)
        
        # Pixel size field
        pixel_size_label = QLabel("Tamanho pixel:")
        pixel_size_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #333333;
                padding: 5px;
            }
        """)
        first_row.addWidget(pixel_size_label)
        
        # Pixel size input field
        self.pixel_size_input = QLabel("0.000")  # Will be updated when image is loaded
        self.pixel_size_input.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #333333;
                padding: 5px;
                min-width: 60px;
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
        """)
        self.pixel_size_input.setAlignment(Qt.AlignCenter)
        first_row.addWidget(self.pixel_size_input)
        
        first_row.addStretch()  # Add stretch to push controls to the left
        
        # Second row: Undo, Redo, Merge, and Mode selection
        second_row = QHBoxLayout()
        
        # Undo button
        self.undo_button = QPushButton("Desfazer")
        self.undo_button.setFixedSize(80, 40)
        self.undo_button.setStyleSheet("""
            QPushButton {
                background-color: #FF5722;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #E64A19;
            }
            QPushButton:pressed {
                background-color: #D84315;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.undo_button.setEnabled(False)  # Disabled until there are operations to undo
        self.undo_button.clicked.connect(self.undo_last_operation)
        second_row.addWidget(self.undo_button)
        
        # Add some spacing
        second_row.addSpacing(20)
        
        # Redo button
        self.redo_button = QPushButton("Refazer")
        self.redo_button.setFixedSize(80, 40)
        self.redo_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.redo_button.setEnabled(False)  # Disabled until there are operations to redo
        self.redo_button.clicked.connect(self.redo_last_operation)
        second_row.addWidget(self.redo_button)
        
        # Add some spacing
        second_row.addSpacing(20)
        
        # Merge button
        self.merge_button = QPushButton("Juntar regiões")
        self.merge_button.setFixedSize(140, 40)
        self.merge_button.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
            QPushButton:pressed {
                background-color: #6A1B9A;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.merge_button.setEnabled(False)  # Disabled until segmentations are selected
        self.merge_button.clicked.connect(self.merge_selected_segmentations)
        second_row.addWidget(self.merge_button)
        
        # Add some spacing
        second_row.addSpacing(20)
        
        # Mode selection radio buttons
        mode_layout = QHBoxLayout()
        
        # Mode label
        mode_label = QLabel("Modo:")
        mode_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #333333;
                padding: 5px;
            }
        """)
        mode_layout.addWidget(mode_label)
        
        # Radio button group
        self.mode_group = QButtonGroup()
        
        # Drag mode radio button
        self.drag_radio = QRadioButton("Mover")
        self.drag_radio.setStyleSheet("""
            QRadioButton {
                font-size: 14px;
                color: #333333;
                padding: 5px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
            }
            QRadioButton::indicator:unchecked {
                border: 2px solid #cccccc;
                background-color: white;
                border-radius: 8px;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #4CAF50;
                background-color: #4CAF50;
                border-radius: 8px;
            }
        """)
        self.drag_radio.setEnabled(False)
        self.mode_group.addButton(self.drag_radio)
        mode_layout.addWidget(self.drag_radio)
        
        # Paint mode radio button
        self.paint_radio = QRadioButton("Rotular")
        self.paint_radio.setStyleSheet("""
            QRadioButton {
                font-size: 14px;
                color: #333333;
                padding: 5px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
            }
            QRadioButton::indicator:unchecked {
                border: 2px solid #cccccc;
                background-color: white;
                border-radius: 8px;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #9C27B0;
                background-color: #9C27B0;
                border-radius: 8px;
            }
        """)
        self.paint_radio.setEnabled(False)
        self.mode_group.addButton(self.paint_radio)
        mode_layout.addWidget(self.paint_radio)
        
        # Drawing mode radio button
        self.drawing_radio = QRadioButton("Segmentar")
        self.drawing_radio.setStyleSheet("""
            QRadioButton {
                font-size: 14px;
                color: #333333;
                padding: 5px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
            }
            QRadioButton::indicator:unchecked {
                border: 2px solid #cccccc;
                background-color: white;
                border-radius: 8px;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #ff6b6b;
                background-color: #ff6b6b;
                border-radius: 8px;
            }
        """)
        self.drawing_radio.setEnabled(False)
        self.mode_group.addButton(self.drawing_radio)
        mode_layout.addWidget(self.drawing_radio)
        
        # Select mode radio button
        self.select_radio = QRadioButton("Selecionar")
        self.select_radio.setStyleSheet("""
            QRadioButton {
                font-size: 14px;
                color: #333333;
                padding: 5px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
            }
            QRadioButton::indicator:unchecked {
                border: 2px solid #cccccc;
                background-color: white;
                border-radius: 8px;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #FF5722;
                background-color: #FF5722;
                border-radius: 8px;
            }
        """)
        self.select_radio.setEnabled(False)
        self.mode_group.addButton(self.select_radio)
        mode_layout.addWidget(self.select_radio)
        
        # Remove mode radio button
        self.remove_radio = QRadioButton("Remover")
        self.remove_radio.setStyleSheet("""
            QRadioButton {
                font-size: 14px;
                color: #333333;
                padding: 5px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
            }
            QRadioButton::indicator:unchecked {
                border: 2px solid #cccccc;
                background-color: white;
                border-radius: 8px;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #F44336;
                background-color: #F44336;
                border-radius: 8px;
            }
        """)
        self.remove_radio.setEnabled(False)
        self.mode_group.addButton(self.remove_radio)
        mode_layout.addWidget(self.remove_radio)
        
        second_row.addLayout(mode_layout)
        
        second_row.addStretch()  # Add stretch to push controls to the left
        
        # Add both rows to the top layout
        top_layout.addLayout(first_row)
        top_layout.addLayout(second_row)
        
        main_layout.addLayout(top_layout)
        
        # Bottom section with canvas and right panel
        bottom_layout = QHBoxLayout()
        
        # Left side: canvas and checkbox
        left_layout = QVBoxLayout()
        
        # Image canvas
        self.canvas = ImageCanvas()
        left_layout.addWidget(self.canvas, 1)  # Give canvas more space
        

        

        
        # Slider for mask opacity
        opacity_layout = QHBoxLayout()
        
        # Label for opacity slider
        opacity_label = QLabel("Opacidade da Máscara:")
        opacity_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #333333;
                padding: 5px;
            }
        """)
        opacity_layout.addWidget(opacity_label)
        
        # Opacity slider
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(50)  # Default to 50% opacity
        self.opacity_slider.setEnabled(False)  # Disabled until mask is loaded
        self.opacity_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #f0f0f0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #5c6bc0;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #4CAF50;
                border-radius: 4px;
            }
        """)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        opacity_layout.addWidget(self.opacity_slider)
        
        # Opacity value label
        self.opacity_value_label = QLabel("50%")
        self.opacity_value_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #666666;
                padding: 5px;
                min-width: 30px;
            }
        """)
        opacity_layout.addWidget(self.opacity_value_label)
        
        left_layout.addLayout(opacity_layout)
        
        bottom_layout.addLayout(left_layout, 1)  # Give left side more space
        
        # Right panel
        self.right_panel = RightPanel()
        self.right_panel.set_canvas(self.canvas)  # Pass canvas reference
        self.canvas.set_right_panel(self.right_panel)  # Pass right panel reference to canvas
        self.canvas.set_main_window(self)  # Pass main window reference to canvas
        bottom_layout.addWidget(self.right_panel)
        
        main_layout.addLayout(bottom_layout)
        
        # Connect the mode radio buttons after canvas is created
        self.drawing_radio.toggled.connect(lambda checked: self.canvas.set_mode("drawing") if checked else None)
        self.drag_radio.toggled.connect(lambda checked: self.canvas.set_mode("drag") if checked else None)
        self.paint_radio.toggled.connect(lambda checked: self.canvas.set_mode("paint") if checked else None)
        self.select_radio.toggled.connect(lambda checked: self.canvas.set_mode("select") if checked else None)
        self.remove_radio.toggled.connect(lambda checked: self.canvas.set_mode("remove") if checked else None)
        
        # Setup keyboard shortcuts
        self.setup_shortcuts()
        
    def load_image(self):
        """Handle image loading"""
        # Set default directory to Imagens folder
        default_dir = os.path.join(self.get_base_directory(), "Imagens")
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Selecionar Imagem",
            default_dir,
            "Arquivos de Imagem (*.png *.jpg *.jpeg *.bmp *.gif *.tiff);;Todos os Arquivos (*)"
        )
        
        if file_path:
            self.current_image_path = file_path  # Store the current image path
            self.canvas.load_image(file_path)
            
            # Calculate and set pixel size based on image width
            if self.canvas.original_pixmap:
                image_width = self.canvas.original_pixmap.width()
                pixel_size_mm = self.calculate_pixel_size(image_width)
                self.update_pixel_size_display(pixel_size_mm)
                # Store pixel size for use in statistics
                self.pixel_size_mm = pixel_size_mm
                # Pass pixel size to right panel
                self.right_panel.set_pixel_size(pixel_size_mm)
            
            # Enable mode radio buttons, save button, and zoom controls
            self.drawing_radio.setEnabled(True)
            self.drag_radio.setEnabled(True)
            self.paint_radio.setEnabled(True)
            self.select_radio.setEnabled(True)
            self.remove_radio.setEnabled(True)
            self.drawing_radio.setChecked(False)  # Start with no mode selected
            self.drag_radio.setChecked(True)  # Start with drag mode selected
            self.paint_radio.setChecked(False)
            self.select_radio.setChecked(False)
            self.canvas.set_mode("drag")  # Set canvas to drag mode
            self.save_button.setEnabled(True)
            self.load_mask_button.setEnabled(True)
            self.merge_button.setEnabled(False)  # Will be enabled when segmentations are selected
            
            # Try to load corresponding mask file from Anotacoes folder
            base_path = os.path.splitext(file_path)[0]  # Remove extension
            base_name = os.path.basename(base_path)  # Get just the filename without path
            anotacoes_dir = os.path.join(self.get_base_directory(), "Anotacoes")
            mask_path = os.path.join(anotacoes_dir, f"{base_name}_mask.png")
            
            if os.path.exists(mask_path):
                self.canvas.load_mask(mask_path)
                self.current_mask_path = mask_path  # Store the mask path
                self.opacity_slider.setEnabled(True)
            else:
                # Try alternative mask extensions
                for ext in ['.jpg', '.jpeg', '.bmp', '.gif', '.tiff']:
                    alt_mask_path = os.path.join(anotacoes_dir, f"{base_name}_mask{ext}")
                    if os.path.exists(alt_mask_path):
                        self.canvas.load_mask(alt_mask_path)
                        self.current_mask_path = alt_mask_path  # Store the mask path
                        self.opacity_slider.setEnabled(True)
                        break
                else:
                    # No mask found - create blank mask for drawing
                    self.canvas.create_blank_mask()
                    self.current_mask_path = None  # No mask loaded
                    self.opacity_slider.setEnabled(True)
            
            # Update window title to show loaded files
            self.update_window_title()
    

    
    def on_opacity_changed(self, value):
        """Handle opacity slider changes"""
        if self.canvas:
            # Lock interface during opacity change
            self.canvas.lock_interface("Ajustando opacidade...")
            
            try:
                opacity = value / 100.0  # Convert percentage to 0.0-1.0 range
                self.canvas.set_mask_opacity(opacity)
                self.opacity_value_label.setText(f"{value}%")
                
                # Store the last non-zero opacity value
                if value > 0:
                    self.last_opacity_value = value
            finally:
                self.canvas.unlock_interface()
    
    def on_pen_thickness_changed(self, value):
        """Handle pen thickness slider value change"""
        if self.canvas:
            self.canvas.pen_thickness = value
            self.pen_thickness_label.setText(str(value))
    
    def calculate_pixel_size(self, image_width):
        """Calculate pixel size in mm based on image width"""
        # Return 0 to display ? mm² instead of calculated values
        return 0.0
    
    def update_pixel_size_display(self, pixel_size_mm):
        """Update the pixel size display"""
        self.pixel_size_input.setText(f"{pixel_size_mm:.3f}")
    
    def toggle_mask_overlay(self):
        """Toggle mask overlay with keyboard shortcut (cycles through opacity levels)"""
        if self.canvas and self.opacity_slider.isEnabled():
            current_value = self.opacity_slider.value()
            if current_value == 0:
                self.opacity_slider.setValue(self.last_opacity_value)  # Show at last set opacity
            else:
                self.opacity_slider.setValue(0)   # Hide (0% opacity)
    
    def load_mask(self):
        """Load a mask file manually"""
        if not hasattr(self, 'current_image_path') or not self.current_image_path:
            return
        
        # Set default directory to Anotacoes folder
        default_dir = os.path.join(self.get_base_directory(), "Anotacoes")
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Carregar Máscara",
            default_dir,
            "Arquivos de Imagem (*.png *.jpg *.jpeg *.bmp *.gif *.tiff);;Todos os Arquivos (*)"
        )
        
        if file_path:
            # Load the mask
            self.canvas.load_mask(file_path)
            self.current_mask_path = file_path  # Store the mask path
            self.opacity_slider.setEnabled(True)
            print(f"Mask loaded successfully from: {file_path}")
            
            # Update window title to show loaded files
            self.update_window_title()
    
    def merge_selected_segmentations(self):
        """Merge selected segmentations"""
        if self.canvas:
            self.canvas.merge_selected_segmentations()
    
    def save_mask(self):
        """Save the current mask to a file and also save the labels array"""
        if not self.canvas.mask_pixmap:
            return False
        
        # Set default directory to Anotacoes folder
        default_dir = os.path.join(self.get_base_directory(), "Anotacoes")
        
        # Get the original image path to suggest a mask filename
        suggested_name = "mask.png"
        
        # If we have an original image path, suggest a mask name based on it
        if hasattr(self, 'current_image_path') and self.current_image_path:
            base_name = os.path.basename(os.path.splitext(self.current_image_path)[0])
            suggested_name = f"{base_name}_mask.png"
            # Set the full path in Anotacoes folder
            suggested_path = os.path.join(default_dir, suggested_name)
        else:
            suggested_path = os.path.join(default_dir, suggested_name)
        
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self,
            "Salvar Máscara",
            suggested_path,
            "Arquivos PNG (*.png);;Arquivos JPEG (*.jpg);;Todos os Arquivos (*)"
        )
        
        if file_path:
            # Lock interface during save operation
            self.canvas.lock_interface("Salvando máscara...")
            
            try:
                # Save the mask directly
                success = self.canvas.mask_pixmap.save(file_path)
                if success:
                    print(f"Mask saved successfully to: {file_path}")
                    
                    # Also save the labels array as a .npy file
                    if self.canvas.pixel_labels is not None:
                        # Create the labels file path with the same base name but .npy extension
                        base_path = os.path.splitext(file_path)[0]  # Remove extension
                        labels_path = f"{base_path}_labels.npy"
                        
                        # Convert to int16 and save
                        labels_array = self.canvas.pixel_labels.astype(np.int16)
                        np.save(labels_path, labels_array)
                        print(f"Labels array saved successfully to: {labels_path}")
                    
                    # Store the saved mask path and update title
                    self.current_mask_path = file_path
                    self.canvas.mark_changes_saved()
                    
                    # Update window title to show loaded files
                    self.update_window_title()
                    
                    return True
                else:
                    print(f"Failed to save mask to: {file_path}")
                    return False
            finally:
                self.canvas.unlock_interface()
        
        return False  # User cancelled the save dialog
    

    
    def undo_last_operation(self):
        """Undo the last operation"""
        if self.canvas:
            self.canvas.undo_last_operation()
    
    def redo_last_operation(self):
        """Redo the last operation"""
        if self.canvas:
            self.canvas.redo_last_operation()
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Ctrl+O for load image
        self.load_shortcut = QShortcut(QKeySequence("Ctrl+O"), self)
        self.load_shortcut.activated.connect(self.load_image)
        
        # Ctrl+S for save mask
        self.save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self.save_shortcut.activated.connect(self.save_mask)
        
        # Ctrl+Z for undo
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo_last_operation)
        
        # Ctrl+X for redo
        self.redo_shortcut = QShortcut(QKeySequence("Ctrl+X"), self)
        self.redo_shortcut.activated.connect(self.redo_last_operation)
        
        # M for toggle mask overlay
        self.mask_toggle_shortcut = QShortcut(QKeySequence("M"), self)
        self.mask_toggle_shortcut.activated.connect(self.toggle_mask_overlay)
        
        # Escape for clear all selections
        self.clear_selection_shortcut = QShortcut(QKeySequence("Escape"), self)
        self.clear_selection_shortcut.activated.connect(self.clear_all_selections)
        
        # Ctrl+M for merge selected segmentations
        self.merge_shortcut = QShortcut(QKeySequence("Ctrl+M"), self)
        self.merge_shortcut.activated.connect(self.merge_selected_segmentations)
    
    def clear_all_selections(self):
        """Clear all selected segmentations"""
        if self.canvas:
            self.canvas.clear_selection()
            print("Cleared all selections")
    
    def closeEvent(self, event):
        """Handle window close event - ask to save if there are unsaved changes"""
        if hasattr(self, 'canvas') and self.canvas.has_unsaved_changes():
            # Ask user if they want to save changes
            reply = QMessageBox.question(
                self,
                "Salvar Alterações?",
                "Existem alterações não salvas. Deseja salvar antes de sair?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )
            
            if reply == QMessageBox.Save:
                # Try to save
                if self.save_mask():
                    # Save was successful, allow close
                    event.accept()
                else:
                    # Save was cancelled or failed, don't close
                    event.ignore()
                    return
            elif reply == QMessageBox.Discard:
                # User chose to discard changes, allow close
                event.accept()
            else:  # QMessageBox.Cancel
                # User cancelled, don't close
                event.ignore()
        else:
            # No unsaved changes, allow close
            event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show the main window
    window = MainWindow()
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec_())


if __name__ == "__main__":

    main()
