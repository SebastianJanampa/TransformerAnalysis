# Save this as image_point_selector.py
import tkinter as tk
from tkinter import messagebox, Frame, Canvas, Listbox, Scrollbar, Button
from PIL import Image, ImageTk, ImageDraw, ImageFont

# Constants for drawing points
POINT_RADIUS = 5
POINT_COLOR = "red"
SELECTED_COLOR = "yellow"

class PointSelector:
    """
    A GUI application using Tkinter to select points on a PIL Image object.
    The GUI loop can be stopped to return the points to a calling script.
    """
    def __init__(self, pil_image):
        self.root = tk.Tk()
        self.pil_image = pil_image
        self.points = []
        self.selected_point_index = None

        self.root.title("Image Point Selector")
        
        # Keep a reference to the PhotoImage to prevent garbage collection
        self.tk_image = ImageTk.PhotoImage(self.pil_image)

        main_frame = Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = Canvas(
            main_frame,
            width=self.pil_image.width,
            height=self.pil_image.height,
            cursor="crosshair"
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        control_frame = Frame(main_frame, padx=10, pady=10)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = Listbox(control_frame, width=30, height=25)
        self.listbox.pack(pady=(0, 10))
        
        # --- KEY CHANGE: This button now quits the mainloop ---
        self.done_button = Button(
            control_frame,
            text="Finalize Selections",
            command=self.root.quit  # Stops the mainloop, allows script to continue
        )
        self.done_button.pack(fill=tk.X)
        
        self.canvas.bind("<Button-1>", self.handle_left_click)
        self.canvas.bind("<Button-3>", self.handle_right_click)
        self.canvas.bind("<B1-Motion>", self.handle_drag)
        self.canvas.bind("<ButtonRelease-1>", self.handle_release)

    def get_points(self):
        """Returns the final list of selected point coordinates."""
        return [p['coords'] for p in self.points]

    def _find_point_by_id(self, canvas_id):
        for i, point in enumerate(self.points):
            if point['id'] == canvas_id:
                return i
        return None

    def handle_left_click(self, event):
        overlapping_items = self.canvas.find_overlapping(
            event.x - POINT_RADIUS, event.y - POINT_RADIUS,
            event.x + POINT_RADIUS, event.y + POINT_RADIUS
        )
        point_ids = [p['id'] for p in self.points]
        clicked_point_id = next((item for item in overlapping_items if item in point_ids), None)

        if clicked_point_id:
            self.selected_point_index = self._find_point_by_id(clicked_point_id)
            if self.selected_point_index is not None:
                self.canvas.itemconfig(self.points[self.selected_point_index]['id'], fill=SELECTED_COLOR)
        else:
            self.add_point(event.x, event.y)

    def handle_right_click(self, event):
        overlapping_items = self.canvas.find_overlapping(
            event.x - POINT_RADIUS, event.y - POINT_RADIUS,
            event.x + POINT_RADIUS, event.y + POINT_RADIUS
        )
        point_ids = [p['id'] for p in self.points]
        point_to_delete_id = next((item for item in overlapping_items if item in point_ids), None)

        if point_to_delete_id:
            index_to_delete = self._find_point_by_id(point_to_delete_id)
            if index_to_delete is not None:
                self.remove_point(index_to_delete)

    def handle_drag(self, event):
        if self.selected_point_index is not None:
            point_id = self.points[self.selected_point_index]['id']
            x, y = event.x, event.y
            self.canvas.coords(
                point_id,
                x - POINT_RADIUS, y - POINT_RADIUS,
                x + POINT_RADIUS, y + POINT_RADIUS
            )
            self.points[self.selected_point_index]['coords'] = (x, y)
            self.update_listbox()

    def handle_release(self, event):
        if self.selected_point_index is not None:
            point_id = self.points[self.selected_point_index]['id']
            self.canvas.itemconfig(point_id, fill=POINT_COLOR)
            self.selected_point_index = None
            self.update_listbox()

    def add_point(self, x, y):
        canvas_id = self.canvas.create_oval(
            x - POINT_RADIUS, y - POINT_RADIUS,
            x + POINT_RADIUS, y + POINT_RADIUS,
            fill=POINT_COLOR,
            outline="black",
            width=1
        )
        self.points.append({'coords': (x, y), 'id': canvas_id})
        self.update_listbox()

    def remove_point(self, index):
        point_to_remove = self.points.pop(index)
        self.canvas.delete(point_to_remove['id'])
        self.update_listbox()

    def update_listbox(self):
        self.listbox.delete(0, tk.END)
        for i, point in enumerate(self.points):
            coords = point['coords']
            self.listbox.insert(tk.END, f"Point {i + 1}: ({coords[0]}, {coords[1]})")