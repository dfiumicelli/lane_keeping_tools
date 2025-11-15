#!/usr/bin/env python3
"""
Genera maschera come l'immagine:
   - GIALLO (sinistra) = Classe 2 = Pixel value 2 (discontinua/tratteggiata)
   - BIANCO (destra) = Classe 1 = Pixel value 1 (continua)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import os
from datetime import datetime

class LineDetectionDebugger(Node):
    def __init__(self):
        super().__init__('line_detection_debugger')
    
        # ===== CONFIGURAZIONI COLORI =====
        self.yellow_config = {
            'lower_hsv': np.array([18, 80, 80]),
            'upper_hsv': np.array([32, 255, 255]),
            'name': 'yellow_discontinua',
            'side': 'left',
            'class_value': 2  # Classe 2 = discontinua
        }
        
        self.white_config = {
            'lower_hsv': np.array([0, 0, 230]),
            'upper_hsv': np.array([180, 20, 255]),
            'name': 'white_continua',
            'side': 'right',
            'class_value': 1  # Classe 1 = continua
        }
        
        # ===== PARAMETRI DETECTION =====
        self.min_contour_area = 600
        self.roi_vertical_start = 0.1
        
        # ===== CREAZIONE CARTELLE =====
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_images')
        self.mask_dir = os.path.join(os.path.dirname(__file__), 'mask_images')
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)
        
        self.frame_count = 0
        
        # ===== SOTTOSCRIZIONE =====
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/image_rect/compressed',
            self.image_callback,
            10
        )
        
        self.get_logger().info(' Line Detection Debugger')
        self.get_logger().info(' Genererà maschere ESATTAMENTE come target:')
        self.get_logger().info('  - GIALLO (sx) = Classe 2 (discontinua)')
        self.get_logger().info('  - BIANCO (dx) = Classe 1 (continua)')
        self.get_logger().info(f' Test: {self.test_dir}')
        self.get_logger().info(f' Mask: {self.mask_dir}')
        self.get_logger().info('⌨️  SPACE: Save | Q: Quit')
    
    def image_callback(self, msg):
        try:
            # Decomprimi
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                return
            
            self.frame_count += 1
            height, width = cv_image.shape[:2]
            
            # Detection
            yellow_mask, white_mask = self.detect_lines(cv_image)

            # Pixel value 2 per giallo (discontinua)
            # Pixel value 1 per bianco (continua)
            class_mask = self._create_class_mask(yellow_mask, white_mask, height, width)
            
            # Visualizzazione
            self.show_debug_view(cv_image, yellow_mask, white_mask, class_mask)
            
            # Controllo tastiera
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # SPACE per salvare
                self.save_images(cv_image, class_mask)
                self.get_logger().info(f'✓ Frame salvato!')
            elif key == ord('q'):  # Q per uscire
                raise KeyboardInterrupt
                
        except Exception as e:
            self.get_logger().error(f'Error in callback: {e}')
    
    def detect_lines(self, image):
        """Detection delle linee gialla e bianca"""
        height, width = image.shape[:2]
        
        # Preprocessing
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        
        roi_start_height = int(height * self.roi_vertical_start)
        
        # ===== DETECTION GIALLA (DISCONTINUA) =====
        yellow_mask = cv2.inRange(hsv, self.yellow_config['lower_hsv'],
                                   self.yellow_config['upper_hsv'])
        yellow_mask[:roi_start_height, :] = 0
        yellow_mask[:, int(width*0.6):] = 0  # Solo sinistra
        
        kernel_yellow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel_yellow)
        yellow_mask = cv2.dilate(yellow_mask, kernel_yellow, iterations=1)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN,
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        
        # ===== DETECTION BIANCA (CONTINUA) =====
        white_mask = cv2.inRange(hsv, self.white_config['lower_hsv'],
                                  self.white_config['upper_hsv'])
        white_mask[:roi_start_height, :] = 0
        white_mask[:, :int(width*0.4)] = 0  # Solo destra
        
        kernel_white = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_white)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_white)
        
        return yellow_mask, white_mask
    
    def _create_class_mask(self, yellow_mask, white_mask, height, width):
        """ CREA MASCHERA ESATTAMENTE COME TARGET
        
        Pixel values:
        - 0 = Background (nero)
        - 2 = Classe 2 = Discontinua/Gialla (sinistra)
        - 1 = Classe 1 = Continua/Bianca (destra)
        """
        # Inizializza con background (0)
        class_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Classe 2 (discontinua/gialla) = 2
        # Questo diventerà GIALLO nella visualizzazione
        class_mask[yellow_mask > 0] = 2
        
        # Classe 1 (continua/bianca) = 1
        # Sovrascrive in caso di sovrapposizione (che non dovrebbe esserci)
        class_mask[white_mask > 0] = 1
        
        return class_mask
    
    def show_debug_view(self, image, yellow_mask, white_mask, class_mask):
        """Visualizza con colori TARGET"""
        height, width = image.shape[:2]
        
        # Ridimensiona per visualizzazione
        scale = 0.5
        h = int(height * scale)
        w = int(width * scale)
        
        img_resized = cv2.resize(image, (w, h))
        class_mask_resized = cv2.resize(class_mask, (w, h))
        
        #  COLORA LA MASCHERA COME TARGET
        # Crea immagine a colori
        mask_display = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Pixel = 0: nero (background)
        mask_display[class_mask_resized == 0] = [0, 0, 0]
        
        # Pixel = 2: GIALLO (Classe 2 = discontinua)
        mask_display[class_mask_resized == 2] = [0, 255, 255]
        
        # Pixel = 1: BIANCO (Classe 1 = continua)
        mask_display[class_mask_resized == 1] = [255, 255, 255]
        
        # Info
        info_img = np.ones((h, w, 3), dtype=np.uint8) * 50
        cv2.putText(info_img, f'Frame: {self.frame_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_img, 'TARGET MASK', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(info_img, f'Class 2 = Yellow', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(info_img, f'Class 1 = White', (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(info_img, 'SPACE: Save | Q: Quit', (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Affiancare
        row = np.hstack([img_resized, info_img, mask_display])
        
        cv2.imshow('Target Mask Generation - Original | Info | Target Mask', row)
    
    def save_images(self, image, class_mask):
        """Salva immagine e maschera con PIXEL VALUES CORRETTI"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        
        # Salva immagine originale
        img_file = os.path.join(self.test_dir, f'image_{timestamp}.jpg')
        cv2.imwrite(img_file, image)
        
        #  SALVA MASCHERA CON PIXEL VALUES:
        # 0 = background
        # 1 = continua (bianca)
        # 2 = discontinua (gialla)
        mask_file = os.path.join(self.mask_dir, f'mask_{timestamp}.png')
        cv2.imwrite(mask_file, class_mask)
        
        # Verificazione dei pixel nella maschera
        unique_values = np.unique(class_mask)
        
        self.get_logger().info(f'✓ Saved:')
        self.get_logger().info(f'   {img_file}')
        self.get_logger().info(f'   {mask_file}')
        self.get_logger().info(f'  Pixel values in mask: {list(unique_values)}')
        self.get_logger().info(f'  └─ 0 = Background')
        self.get_logger().info(f'  └─ 1 = Continua (WHITE)')
        self.get_logger().info(f'  └─ 2 = Discontinua (YELLOW)')


def main(args=None):
    rclpy.init(args=args)
    debugger = LineDetectionDebugger()
    try:
        rclpy.spin(debugger)
    except KeyboardInterrupt:
        debugger.get_logger().info('Stopping...')
    finally:
        debugger.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
