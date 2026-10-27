#!/usr/bin/env python3
"""
Práctica 4: Calibración de Cámara
Script de Captura Interactiva con Intel RealSense D435i y ROS 2.
Permite capturar 20 imágenes del tablero de ajedrez mediante la tecla [ESPACIO].
"""

import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Consejos de orientación para capturar un conjunto variado de calibración
TIPS = [
    "1. Tablero centrado, completamente frontal",
    "2. Tablero inclinado ligeramente hacia la izquierda (~20 deg)",
    "3. Tablero inclinado ligeramente hacia la derecha (~20 deg)",
    "4. Tablero inclinado hacia arriba (~20 deg)",
    "5. Tablero inclinado hacia abajo (~20 deg)",
    "6. Tablero en esquina superior izquierda del encuadre",
    "7. Tablero en esquina superior derecha del encuadre",
    "8. Tablero en esquina inferior izquierda del encuadre",
    "9. Tablero en esquina inferior derecha del encuadre",
    "10. Tablero más cerca de la cámara (ocupa 60-70% de la imagen)",
    "11. Tablero más lejos de la cámara (ocupa 25-30% de la imagen)",
    "12. Tablero rotado 45 grados en diagonal horaria",
    "13. Tablero rotado 45 grados en diagonal antihoraria",
    "14. Tablero pegado al borde superior (para captar distorsión)",
    "15. Tablero pegado al borde inferior",
    "16. Tablero pegado al borde izquierdo",
    "17. Tablero pegado al borde derecho",
    "18. Tablero con inclinación combinada (arriba + izquierda)",
    "19. Tablero con inclinación combinada (abajo + derecha)",
    "20. Tablero frontal a media distancia (última verificación)"
]

class CalibrationCaptureNode(Node):
    def __init__(self, target_photos=20, output_dir="imagenes"):
        super().__init__('calibration_capture_node')
        self.bridge = CvBridge()
        self.target_photos = target_photos
        self.output_dir = output_dir
        self.photo_count = 0

        os.makedirs(self.output_dir, exist_ok=True)

        self.latest_frame = None

        # Suscripción al tópico RGB de la RealSense
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.get_logger().info("=" * 60)
        self.get_logger().info(" CAPTURADOR DE FOTOS PARA CALIBRACIÓN (RealSense D435i)")
        self.get_logger().info(f" Destino de imágenes: {os.path.abspath(self.output_dir)}")
        self.get_logger().info(f" Meta: {self.target_photos} fotos")
        self.get_logger().info(" Esperando tópico: /camera/camera/color/image_raw...")
        self.get_logger().info("=" * 60)

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error procesando frame: {e}")

    def run(self):
        window_title = 'Calibracion RealSense D435i - [ESPACIO: Disparar | Q: Salir]'
        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

        while rclpy.ok() and self.photo_count < self.target_photos:
            rclpy.spin_once(self, timeout_sec=0.03)

            if self.latest_frame is None:
                continue

            frame = self.latest_frame.copy()
            h, w = frame.shape[:2]

            # Detección rápida de esquinas en tiempo real para retroalimentación
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Intentar detectar cuadrícula 8x6 (o 7x5)
            pattern_found = False
            corners_detected = None
            grid_size = (8, 6)
            
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners = cv2.findChessboardCorners(gray, grid_size, flags)
            if ret:
                pattern_found = True
                corners_detected = corners
            else:
                # Probar también con 7x5 por si el usuario tiene otro patrón
                ret2, corners2 = cv2.findChessboardCorners(gray, (7, 5), flags)
                if ret2:
                    pattern_found = True
                    corners_detected = corners2
                    grid_size = (7, 5)

            # Barra superior con fondo translúcido
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 75), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            # Texto de estado
            status_text = f"Foto: {self.photo_count + 1}/{self.target_photos}   |   [ESPACIO/C]: Tomar Foto   |   [Q]: Salir"
            cv2.putText(frame, status_text, (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

            # Sugerencia de pose actual
            tip_idx = min(self.photo_count, len(TIPS) - 1)
            cv2.putText(frame, f"Sugerencia: {TIPS[tip_idx]}", (15, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 215, 0), 2, cv2.LINE_AA)

            # Dibujar esquinas y estado
            if pattern_found:
                cv2.drawChessboardCorners(frame, grid_size, corners_detected, True)
                cv2.circle(frame, (w - 30, 35), 14, (0, 255, 0), -1)
                cv2.putText(frame, "TABLERO DETECTADO [OK]", (w - 240, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.circle(frame, (w - 30, 35), 14, (0, 140, 255), -1)
                cv2.putText(frame, "Enfoca el tablero", (w - 180, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1, cv2.LINE_AA)

            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xFF

            # Capturar con barra espaciadora o tecla C
            if key in [ord('c'), ord('C'), 32]:
                self.photo_count += 1
                filename = f"foto_{self.photo_count:02d}.png"
                filepath = os.path.join(self.output_dir, filename)
                
                # Guardamos el frame original sin superposiciones de texto
                cv2.imwrite(filepath, self.latest_frame)
                self.get_logger().info(f"[OK] Captura {self.photo_count}/{self.target_photos} guardada en: {filepath}")

                # Efecto flash de obturador
                flash = np.full_like(frame, 255)
                cv2.imshow(window_title, flash)
                cv2.waitKey(80)

            # Salir con Q o ESC
            elif key in [ord('q'), ord('Q'), 27]:
                self.get_logger().info("Captura interrumpida por el usuario.")
                break

        if self.photo_count >= self.target_photos:
            self.get_logger().info("=" * 60)
            self.get_logger().info(f" ¡FELICITACIONES! Se capturaron las {self.target_photos} fotos exitosamente.")
            self.get_logger().info(f" Ubicación: {os.path.abspath(self.output_dir)}")
            self.get_logger().info(" Ahora ejecuta: python3 calibracion_completa.py")
            self.get_logger().info("=" * 60)

        cv2.destroyAllWindows()

def main():
    rclpy.init()
    node = CalibrationCaptureNode(
        target_photos=20,
        output_dir="/home/maykoll_vanegas/Escritorio/Practica4_Calibracion/imagenes"
    )
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
