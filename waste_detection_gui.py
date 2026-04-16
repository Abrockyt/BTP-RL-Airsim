import tkinter as tk
from tkinter import scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import airsim
import threading
import time
import torch
import nest_asyncio

# Apply nest_asyncio to prevent Tornado IOLoop "already running" errors
# from AirSim when wrapping in a background Tkinter daemon.
nest_asyncio.apply()

class WasteDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("River Waste Detection UAV Dashboard")
        self.root.geometry("900x800")
        self.root.configure(bg="#2b2b2b")
        
        self.vehicle_name = "Drone1"
        self.is_running = True
        self.patrolling = False
        self.global_shape_counts = {
            "Triangle": 0,
            "Square Box": 0,
            "Rectangle": 0,
            "Circle/Sphere": 0,
            "Cylinder": 0,
            "Waste (Unclassified)": 0
        }
        self.active_tracker = []
        self.inspecting = False        # true when dropping altitude to investigate
        self.is_zoomed_in = False      # true when currently hovering at low altitude for shape confirmation
        
        self.verified_wastes = []
        self.drone_path = []
        self.waste_id_counter = 1
        self.frame_count = 0
        
        # UI Setup
        self.setup_ui()

        # Load YOLOv8
        self.log_message("Loading YOLOv8 model...")
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt') # Lightweight fast model
            self.log_message("YOLOv8 model loaded successfully.")
        except Exception as e:
            self.log_message(f"Failed to load YOLOv8: {e}")
            self.log_message("Falling back to raw RGB feed.")
        
        # AirSim Connection
        self.client = None
        self.connect_airsim()
        
        # Start Flight Loop
        self.flight_thread = threading.Thread(target=self.flight_loop, daemon=True)
        self.flight_thread.start()

    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#2b2b2b")
        header_frame.pack(pady=10, fill=tk.X)
        
        title = tk.Label(header_frame, text="RIVER WASTE DETECTION DASHBOARD", font=("Arial", 16, "bold"), bg="#2b2b2b", fg="white")
        title.pack()
        
        self.warning_label = tk.Label(header_frame, text="🚨 POLLUTION ALERT: High Waste Concentration 🚨", font=("Arial", 14, "bold"), bg="#2b2b2b", fg="#2b2b2b")
        self.warning_label.pack(pady=5)
        
        # Camera Feeds
        cam_frame = tk.Frame(self.root, bg="#2b2b2b")
        cam_frame.pack(pady=5)
        
        self.rgb_label = tk.Label(cam_frame, bg="black", width=400, height=250)
        self.rgb_label.grid(row=0, column=0, padx=10)
        tk.Label(cam_frame, text="RGB + YOLO Feed", bg="#2b2b2b", fg="white").grid(row=1, column=0)
        
        self.depth_label = tk.Label(cam_frame, bg="black", width=400, height=250)
        self.depth_label.grid(row=0, column=1, padx=10)
        tk.Label(cam_frame, text="Depth Feed", bg="#2b2b2b", fg="white").grid(row=1, column=1)

        # Waste Shape Detection Table
        from tkinter import ttk
        table_frame = tk.Frame(self.root, bg="#2b2b2b")
        table_frame.pack(pady=5, padx=20, fill=tk.X)
        
        style = ttk.Style()
        style.configure("Treeview.Heading", font=('Arial', 10, 'bold'))
        style.configure("Treeview", font=('Arial', 10), rowheight=25)
        
        self.shape_table = ttk.Treeview(table_frame, columns=("ID", "Type", "Location", "Quantity"), show="headings", height=5)
        self.shape_table.heading("ID", text="ID")
        self.shape_table.heading("Type", text="Detection Type")
        self.shape_table.heading("Location", text="Location (X,Y)")
        self.shape_table.heading("Quantity", text="Quantity")
        self.shape_table.column("ID", width=50, anchor=tk.CENTER)
        self.shape_table.column("Type", width=150, anchor=tk.CENTER)
        self.shape_table.column("Location", width=150, anchor=tk.CENTER)
        self.shape_table.column("Quantity", width=100, anchor=tk.CENTER)
        self.shape_table.pack(fill=tk.X)
        
        # Telemetry
        telemetry_frame = tk.Frame(self.root, bg="#3b3b3b", bd=2, relief=tk.SUNKEN)
        telemetry_frame.pack(pady=10, fill=tk.X, padx=20)
        
        self.alt_var = tk.StringVar(value="Altitude: 0.0 m")
        self.speed_var = tk.StringVar(value="Speed: 0.0 m/s")
        self.waste_var = tk.StringVar(value="Waste Detected: 0")
        
        tk.Label(telemetry_frame, textvariable=self.alt_var, font=("Arial", 12), bg="#3b3b3b", fg="white").pack(side=tk.LEFT, padx=30, pady=10)
        tk.Label(telemetry_frame, textvariable=self.speed_var, font=("Arial", 12), bg="#3b3b3b", fg="white").pack(side=tk.LEFT, padx=30, pady=10)
        tk.Label(telemetry_frame, textvariable=self.waste_var, font=("Arial", 12, "bold"), bg="#3b3b3b", fg="orange").pack(side=tk.RIGHT, padx=30, pady=10)
        
        # Controls
        controls_frame = tk.Frame(self.root, bg="#2b2b2b")
        controls_frame.pack(pady=10)
        
        btn_style = {"font": ("Arial", 12, "bold"), "width": 15, "pady": 5, "fg": "white"}
        
        tk.Button(controls_frame, text="Deploy & Patrol", bg="#28a745", command=self.deploy_and_patrol, **btn_style).grid(row=0, column=0, padx=10)
        tk.Button(controls_frame, text="Stop / Land", bg="#dc3545", command=self.land, **btn_style).grid(row=0, column=1, padx=10)
        
        # System Log
        log_frame = tk.Frame(self.root, bg="#2b2b2b")
        log_frame.pack(pady=10, fill=tk.BOTH, expand=True, padx=20)
        
        tk.Label(log_frame, text="System Log", bg="#2b2b2b", fg="white", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.log_box = scrolledtext.ScrolledText(log_frame, height=8, bg="#1e1e1e", fg="lightgreen", font=("Consolas", 10))
        self.log_box.pack(fill=tk.BOTH, expand=True)

    def connect_airsim(self):
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
            
            # Use asyncio-safe AirSim calls for taking off when inside a threading loop
            import __main__
            import nest_asyncio
            nest_asyncio.apply()
            
            self.log_message(f"Connected to AirSim. Controlled vehicle: {self.vehicle_name}")
        except Exception as e:
            self.client = None
            self.log_message(f"AirSim Connection Error: {e}")

    def log_message(self, msg, color=None):
        timestamp = time.strftime("%H:%M:%S")
        self.log_box.insert(tk.END, f"[{timestamp}] {msg}\n")
        if color:
            # simple tagging for color
            line_end = self.log_box.index("end-1c")
            line_start = f"{float(line_end) - 1.0}"
            tag_name = f"color_{color}"
            self.log_box.tag_add(tag_name, line_start, line_end)
            self.log_box.tag_config(tag_name, foreground=color)
            
        self.log_box.see(tk.END)

    def deploy_and_patrol(self):
        self.global_shape_counts = {k: 0 for k in self.global_shape_counts}
        self.active_tracker = []
        self.verified_wastes = []
        self.drone_path = []
        self.waste_id_counter = 1
        self.frame_count = 0
        self.inspecting = False
        self.is_zoomed_in = False
        
        self.log_message("Initiating deployment sequence...")
        self.patrolling = True
        def _task():
            try:
                c = airsim.MultirotorClient()
                c.confirmConnection()
                c.enableApiControl(True, vehicle_name=self.vehicle_name)
                c.armDisarm(True, vehicle_name=self.vehicle_name)
                
                self.log_message("Taking off...")
                c.takeoffAsync(vehicle_name=self.vehicle_name).join()
                
                # Fly to a high patrol altitude to spot areas of interest
                state = c.getMultirotorState(vehicle_name=self.vehicle_name)
                z = state.kinematics_estimated.position.z_val
                target_z = z - 12.0  
                
                self.log_message("Ascending to stable high patrol altitude (12m up)...")
                c.moveToZAsync(target_z, 5.0, vehicle_name=self.vehicle_name).join()
                
                # Command a hover to stabilize before moving
                c.hoverAsync(vehicle_name=self.vehicle_name).join()
                
                self.log_message("Starting river patrol... Moving forward very slowly & stably.")
                # moveByVelocityZAsync ensures it never drops down, maintaining perfect height!
                # We use ForwardOnly drivetrain and fixed YawMode to keep the drone completely steady without side-drifting
                c.moveByVelocityZAsync(1.5, 0, target_z, 3000, 
                                       drivetrain=airsim.DrivetrainType.ForwardOnly, 
                                       yaw_mode=airsim.YawMode(False, 0),
                                       vehicle_name=self.vehicle_name)
            except Exception as e:
                print("Deployment error:", e)
                self.log_message(f"Deployment error: {e}", color="red")
        threading.Thread(target=_task, daemon=True).start()

    def inspect_objects(self):
        """Halts the drone and descends to get a clear visual confirmation of the object's shape."""
        self.inspecting = True
        self.log_message("Alert: Suspicious objects detected!", color="orange")
        
        def _inspect_task():
            try:
                c = airsim.MultirotorClient()
                c.confirmConnection()
                
                # Stop moving forward
                c.moveByVelocityAsync(0, 0, 0, 1, vehicle_name=self.vehicle_name).join()
                
                self.log_message("Halting and dropping altitude to confirm shape...", color="orange")
                
                state = c.getMultirotorState(vehicle_name=self.vehicle_name)
                current_z = state.kinematics_estimated.position.z_val
                
                # We removed the zoom inspection entirely so it flies non-stop!
                pass
                
            except Exception as e:
                self.log_message(f"Inspection error: {e}", color="red")
            finally:
                self.inspecting = False

        threading.Thread(target=_inspect_task, daemon=True).start()

    def land(self):
        self.log_message("Stopping and initiating landing...")
        self.patrolling = False
        
        # Show summary instantly instead of waiting for the slow landing animation
        self.show_mission_summary()
        
        def _task():
            try:
                c = airsim.MultirotorClient()
                c.confirmConnection()
                c.enableApiControl(True, vehicle_name=self.vehicle_name)
                # Force instant stop
                c.moveByVelocityAsync(0, 0, 0, 1, vehicle_name=self.vehicle_name).join()
                
                # Drop altitude faster manually before landing command
                state = c.getMultirotorState(vehicle_name=self.vehicle_name)
                z = state.kinematics_estimated.position.z_val
                # Move to 2m above ground fast (assuming ground is ~0 in Z)
                if z < -3.0:
                    c.moveToZAsync(-2.0, 5.0, vehicle_name=self.vehicle_name).join()
                
                c.landAsync(vehicle_name=self.vehicle_name).join()
                c.armDisarm(False, vehicle_name=self.vehicle_name)
                
            except Exception as e:
                print("Land error:", e)
                self.log_message(f"Land error: {e}", color="red")
        threading.Thread(target=_task, daemon=True).start()

    def show_mission_summary(self):
        summary_win = tk.Toplevel(self.root)
        summary_win.title("Mission Complete - Waste Summary")
        summary_win.geometry("900x700")
        summary_win.configure(bg="#2b2b2b")
        
        tk.Label(summary_win, text="🏆 Patrol Complete: Total Waste Detected", fg="white", bg="#2b2b2b", font=("Arial", 16, "bold")).pack(pady=15)
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            fig = Figure(figsize=(7, 5), dpi=100)
            ax = fig.add_subplot(111)
            ax.set_title("Drone Flight Path & Waste Map")
            ax.set_xlabel("X (World Coordinates)")
            ax.set_ylabel("Y (World Coordinates)")
            ax.grid(True)
            
            # Plot Drone Path
            if self.drone_path:
                px = [p[0] for p in self.drone_path]
                py = [p[1] for p in self.drone_path]
                ax.plot(px, py, 'c--', alpha=0.6, label="Flight Path")
                if px:
                    ax.plot(px[0], py[0], 'go', label="Start")
                    ax.plot(px[-1], py[-1], 'ro', label="End")
            
            colors = ['m', 'y', 'g', 'r', 'b', 'orange', 'cyan']
            shape_color_map = {}
            color_index = 0
            
            # Use clustered array if computed, otherwise fallback
            for item in self.shape_table.get_children():
                vals = self.shape_table.item(item)['values']
                # vals = (ID, Type, Location. Quantity)
                s = vals[1] # shape
                loc = vals[2] # "X:..., Y:..."
                
                import re
                try:
                    # extract floats
                    nums = re.findall(r"[-+]?\d*\.\d+|\d+", loc)
                    wx, wy = float(nums[0]), float(nums[1])
                except:
                    continue
                    
                qty = int(vals[3])
                
                if s not in shape_color_map:
                    shape_color_map[s] = colors[color_index % len(colors)]
                    color_index += 1
                    
                # Bigger dot for bigger clusters
                ax.scatter(wx, wy, color=shape_color_map[s], s=100 + (qty * 20), zorder=5, label=s if s not in ax.get_legend_handles_labels()[1] else "")
                ax.annotate(f"{qty}x" if qty > 1 else f"ID:{vals[0]}", (wx+0.5, wy+0.5), fontsize=8, zorder=6)
                
            ax.legend()
            
            canvas = FigureCanvasTkAgg(fig, master=summary_win)
            canvas.draw()
            canvas.get_tk_widget().pack(pady=10, fill=tk.BOTH, expand=True)
            
        except ImportError:
            tk.Label(summary_win, text="*(matplotlib not installed. Cannot render map.)*", fg="red", bg="#2b2b2b").pack(pady=5)
            
        grand_total = len(self.verified_wastes)
        tk.Label(summary_win, text=f"Total Unique Objects Tracked: {grand_total}", fg="#28a745", bg="#2b2b2b", font=("Arial", 14, "bold")).pack(pady=10)

    def update_image(self, label, cv_img):
        # Convert cv2 image to PIL format for Tkinter
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_img)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        label.configure(image=tk_img)
        label.image = tk_img

    def flight_loop(self):
        alert_active = False
        alert_timer = 0
        
        # Instantiate a dedicated client for this background thread
        # to prevent Tornado IOLoop conflicts with the main thread
        try:
            loop_client = airsim.MultirotorClient()
            loop_client.confirmConnection()
        except:
            print("Flight loop failed to connect to AirSim")
            return
            
        while self.is_running:
            try:
                # 1. Get Telemetry
                state = loop_client.getMultirotorState(vehicle_name=self.vehicle_name)
                z = -state.kinematics_estimated.position.z_val
                vx = state.kinematics_estimated.linear_velocity.x_val
                vy = state.kinematics_estimated.linear_velocity.y_val
                speed = np.sqrt(vx**2 + vy**2)
                
                self.alt_var.set(f"Altitude: {z:.2f} m")
                self.speed_var.set(f"Speed: {speed:.2f} m/s")

                self.frame_count += 1
                do_yolo = (self.frame_count % 3 == 0)
                
                x_val = state.kinematics_estimated.position.x_val
                y_val = state.kinematics_estimated.position.y_val
                if not self.drone_path or np.hypot(self.drone_path[-1][0] - x_val, self.drone_path[-1][1] - y_val) > 0.5:
                    self.drone_path.append((x_val, y_val))

                # 2. Get Images using DepthPlanar to avoid sphere distortion
                requests = [
                    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)
                ]
                responses = loop_client.simGetImages(requests, vehicle_name=self.vehicle_name)
                
                if len(responses) == 2:
                    # Process RGB Image
                    rgb_response = responses[0]
                    img_1d = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
                    if len(img_1d) > 0:
                        # AirSim returns BGRA or BGR if requested Scene, usually 3 channels BGR
                        # But wait, image_data_uint8 is often 3 channels (BGR)
                        # Let's verify shape and convert cautiously
                        img_bgr = img_1d.reshape(rgb_response.height, rgb_response.width, 3)
                        
                        # Resize for GUI
                        img_bgr = cv2.resize(img_bgr, (400, 250))
                        
                        # YOLO expects RGB
                        img_rgb_for_yolo = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        
                        object_count = 0
                        
                        # 1. AI YOLOv8 Detection (for real-world object shapes if any)
                        if self.model and do_yolo:
                            results = self.model(img_rgb_for_yolo, verbose=False)
                            
                            # YOLOv8 Results parsing
                            for r in results:
                                boxes = r.boxes
                                for box in boxes:
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                    conf = float(box.conf[0])
                                    cls = int(box.cls[0])
                                    label = self.model.names[cls]
                                    
                                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 165, 0), 2) # Orange for AI
                                    cv2.putText(img_bgr, f"AI:{label} {conf:.2f}", (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
                                    object_count += 1
                        
                    # We significantly lower the threshold from 0.2 meters to just 0.05 meters (5cm).
                    # This guarantees even half-submerged small blocks trigger the mask, 
                    # while > 0.5 ignores the black lens corners.
                    
                    depth_response = responses[1]
                    depth_array = np.array(depth_response.image_data_float, dtype=np.float32)
                    
                    if len(depth_array) > 0 and depth_response.height > 0 and depth_response.width > 0:
                        depth_array = depth_array.reshape(depth_response.height, depth_response.width)
                        
                        # A) Pure Depth Filter: Outliers closer than the flat water surface 
                        water_depth = np.median(depth_array)
                        
                        depth_mask = np.where((depth_array < (water_depth - 0.15)) & (depth_array > 0.8), 255, 0).astype(np.uint8)
                        depth_mask = cv2.resize(depth_mask, (400, 250))
                        
                        # Clean up the depth mask with morphology to make it highly accurate
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                        depth_mask = cv2.erode(depth_mask, kernel, iterations=1)
                        depth_mask = cv2.dilate(depth_mask, kernel, iterations=2)
                        
                        # Shave off the extreme corners so the mask strictly focuses on the internal frame
                        margin = 25
                        depth_mask[0:margin, :] = 0
                        depth_mask[-margin:, :] = 0
                        depth_mask[:, 0:margin] = 0
                        depth_mask[:, -margin:] = 0

                        object_count = 0
                        shape_counts = {
                            "Triangle": 0,
                            "Square Box": 0,
                            "Rectangle": 0,
                            "Circle/Sphere": 0,
                            "Cylinder": 0,
                            "Waste (Unclassified)": 0
                        }
                        
                        # Find contours strictly on the Depth Mask
                        contours, _ = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # Prevent multiple objects from claiming the same tracker in a single frame
                        used_tracker_indices = set()
                        
                        for cnt in contours:
                            area = cv2.contourArea(cnt)
                            
                            # FAST DETECTION BEHAVIOR:
                            if 15 < area < 40000:
                                peri = cv2.arcLength(cnt, True)
                                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                                x, y, w, h = cv2.boundingRect(approx)
                                aspect_ratio = float(w)/h
                                
                                # Name shape based on geometry
                                shape_name = "Waste (Unclassified)"
                                if len(approx) == 3:
                                    shape_name = "Triangle"
                                elif len(approx) == 4:
                                    if 0.85 <= aspect_ratio <= 1.15:
                                        shape_name = "Square Box"
                                    else:
                                        shape_name = "Rectangle"
                                else:
                                    circularity = 4 * np.pi * (area / (peri * peri)) if peri > 0 else 0
                                    if circularity > 0.75:
                                        shape_name = "Circle/Sphere"
                                    else:
                                        shape_name = "Cylinder"
                                        
                                cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                cv2.putText(img_bgr, f"{shape_name}", (x, max(y-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                                import math
                                # Extremely strict tracking: match must be within 40 pixels, preventing grouped items resolving as 1
                                cx, cy = x + w//2, y + h//2
                                matched = False
                                for i, (tx, ty, tshape, frames_seen, confirmed) in enumerate(self.active_tracker):
                                    if i not in used_tracker_indices and math.hypot(tx - cx, ty - cy) < 40:
                                        new_frames = frames_seen + 1
                                        new_confirmed = confirmed
                                        
                                        if new_frames >= 3 and not confirmed:
                                            new_confirmed = True
                                            
                                            w_id = self.waste_id_counter
                                            self.waste_id_counter += 1
                                            
                                            drone_x = state.kinematics_estimated.position.x_val
                                            drone_y = state.kinematics_estimated.position.y_val
                                            
                                            self.verified_wastes.append({
                                                'id': w_id,
                                                'shape': shape_name,
                                                'x': drone_x,
                                                'y': drone_y,
                                                'qty': 1
                                            })
                                            self.log_message(f"DETECTED: [ID {w_id}] {shape_name} at X:{drone_x:.1f}, Y:{drone_y:.1f}", color="green")
                                            
                                        self.active_tracker[i] = (cx, cy, shape_name, new_frames, new_confirmed)
                                        used_tracker_indices.add(i)
                                        matched = True
                                        break
                                
                                if not matched:
                                    self.active_tracker.append((cx, cy, shape_name, 1, False))
                                
                                object_count += 1
                                    
                        # Update the shape table rows
                        for row in self.shape_table.get_children():
                            self.shape_table.delete(row)
                            
                        # Here we cluster objects that are close to each other into a "Bunch" for the GUI table and map
                        clustered_wastes = []
                        if len(self.verified_wastes) > 0:
                            processed = set()
                            for idx, w1 in enumerate(self.verified_wastes):
                                if idx in processed:
                                    continue
                                
                                group_shapes = {w1['shape']: 1}
                                qty = 1
                                
                                group_x_m = w1['x']
                                group_y_m = w1['y']
                                processed.add(idx)
                                
                                # Find neighbors
                                for idx2, w2 in enumerate(self.verified_wastes):
                                    if idx2 not in processed:
                                        import math
                                        if math.hypot(w1['x'] - w2['x'], w1['y'] - w2['y']) < 5.0: # Close proximity grouping (AirSim meters)
                                            qty += 1
                                            group_shapes[w2['shape']] = group_shapes.get(w2['shape'], 0) + 1
                                            processed.add(idx2)
                                            group_x_m = (group_x_m + w2['x'])/2 # roughly center
                                            group_y_m = (group_y_m + w2['y'])/2
                                            
                                if qty > 1:
                                    majority_shape = max(group_shapes, key=group_shapes.get)
                                    name = f"Cluster ({majority_shape})" if len(group_shapes) == 1 else "Mixed Cluster"
                                    clustered_wastes.append({'id': w1['id'], 'shape': name, 'x':group_x_m, 'y':group_y_m, 'qty':qty})
                                else:
                                    clustered_wastes.append({'id': w1['id'], 'shape': w1['shape'], 'x':group_x_m, 'y':group_y_m, 'qty':1})
                            
                        # Insert verified specific items into table
                        for w in (clustered_wastes if 'clustered_wastes' in locals() else self.verified_wastes):
                            self.shape_table.insert("", tk.END, values=(w['id'], w['shape'], f"X:{w['x']:.1f}, Y:{w['y']:.1f}", w['qty']))
                                
                        self.waste_var.set(f"Waste Detected: {object_count}")
                        self.update_image(self.rgb_label, img_bgr)
                        
                        # Alert Logic
                        if object_count > 0:
                            if not alert_active:
                                self.log_message(f"🚨 ALERT: Depth-Detected Object!", color="red")
                                self.warning_label.config(fg="red")
                                alert_active = True
                            alert_timer = 20
                        else:
                            if alert_active:
                                alert_timer -= 1
                                if alert_timer <= 0:
                                    self.warning_label.config(fg="#2b2b2b")
                                    alert_active = False

                        # Update Depth Visualization
                        min_d = max(0, water_depth - 5.0)
                        max_d = water_depth + 1.0          
                        
                        depth_clipped = np.clip(depth_array, min_d, max_d)
                        depth_norm = cv2.normalize(depth_clipped, None, 255, 0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                        depth_color = cv2.resize(depth_color, (400, 250))
                        
                        # Draw the same bounding boxes on the Depth Feed view
                        for cnt in contours:
                            area = cv2.contourArea(cnt)
                            if 10 < area < 5000:
                                x, y, w, h = cv2.boundingRect(cnt)
                                cv2.rectangle(depth_color, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                
                        self.update_image(self.depth_label, depth_color)

            except Exception as e:
                # Silently catch brief connection errors during loop
                print(f"Flight loop error: {e}")
                
            time.sleep(0.1) # 10 Hz refresh rate

    def on_close(self):
        self.is_running = False
        if self.client:
            self.client.enableApiControl(False, vehicle_name=self.vehicle_name)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = WasteDetectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
