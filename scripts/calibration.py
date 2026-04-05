#!/usr/bin/env python

from tkinter import *
from tkinter import ttk
from tkinter import messagebox as tkMessageBox
from tkinter import filedialog as tkFileDialog

import queue as Queue

from PIL import ImageTk, Image

import cv2
import numpy as np
import threading
from threading import Lock
import sys
import argparse


def mat2tk(img_mat, size=None):
    if size is None:
        pil_img = Image.fromarray(img_mat)
    else:
        src_width = img_mat.shape[1]
        dst_width = size[0]
        assert isinstance(dst_width, int)
        step = max(1, src_width // dst_width)
        pil_img = Image.fromarray(img_mat[::step, ::step])

    return ImageTk.PhotoImage(pil_img)


def find_chess_board(img, pattern_size):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCorners(img_gray, pattern_size)
    img_chess = None
    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(img_gray, corners, (5, 5), (-1, -1), term)
        img_chess = img.copy()
        cv2.drawChessboardCorners(img_chess, pattern_size, corners, found)

    return found, img_chess


class CalibrationApp:
    def __init__(self, device, width=1920, height=1080, square_size=1., pattern_size=(8, 6)):
        self.device = device
        self.width = width
        self.height = height
        self.square_size = square_size
        self.pattern_size = pattern_size # Store pattern size

        self.capture_lock = Lock()
        self.capture = None
        self.input_frame = None
        self.frames = []
        self.frames_thumb = []
        self.frames_tiny = []
        self.cur_frame_idx = 0
        self.K = None
        self.dist_coefs = None
        self.rms = 0
        self.size_msg = 'Unknown size.'
        self.start_capture()

        self.tk_root = Tk()
        self.tk_root.title('Camera calibration')

        self.main_frame = ttk.Frame(self.tk_root, padding="3 3 12 12")
        self.main_frame.grid(column=0, row=0, sticky=(N, W, E, S))
        
        thumb_w = 320
        thumb_h = int(float(thumb_w) / self.width * self.height)
        self.thumb_size = (thumb_w, thumb_h)

        # Setup GUI Elements
        pil_img = Image.new('RGB', self.thumb_size)
        self.blank = ImageTk.PhotoImage(pil_img)
        live_frame = ttk.Labelframe(self.main_frame, text='Live camera')
        live_frame.grid(column=0, row=0, sticky='NWES', padx=(3, 3))

        self.in_frame_label = ttk.Label(live_frame, image=self.blank)
        self.in_frame_label.grid(column=0, row=0, padx=11, pady=11)

        self.size_label = ttk.Label(live_frame, text=self.size_msg)
        self.size_label.grid(column=0, row=1, padx=11, sticky=W)

        b = ttk.Button(live_frame, text="Add", command=self.add_frame)
        b.grid(column=0, row=2, padx=3, pady=5)

        browser_frame = ttk.Labelframe(self.main_frame, text='Selected frames')
        browser_frame.grid(column=1, row=0, rowspan=5, sticky='NWES', padx=(3, 3))

        tiny_size = (thumb_w // 2, thumb_h // 2)
        pil_img_tiny = Image.new('RGB', tiny_size)
        self.blank_sthumb = ImageTk.PhotoImage(pil_img_tiny)
        
        self.left_flabel = ttk.Label(browser_frame, image=self.blank_sthumb, compound=TOP)
        self.left_flabel.grid(column=0, row=0, padx=11, pady=11)

        self.center_flabel = ttk.Label(browser_frame, image=self.blank, compound=TOP)
        self.center_flabel.grid(column=1, row=0, columnspan=3, padx=11, pady=11)

        self.right_flabel = ttk.Label(browser_frame, image=self.blank_sthumb, compound=TOP)
        self.right_flabel.grid(column=4, row=0, padx=11, pady=11)

        ttk.Button(browser_frame, text="< Previous", command=self.prev_frame).grid(column=1, row=1, padx=3, pady=5)
        ttk.Button(browser_frame, text="Remove", command=self.remove_frame).grid(column=2, row=1, padx=3, pady=5)
        ttk.Button(browser_frame, text="Next >", command=self.next_frame).grid(column=3, row=1, padx=3, pady=5)

        cal_frame = ttk.Labelframe(self.main_frame, text='Calibration')
        cal_frame.grid(column=1, row=5, sticky='NWES', padx=(3, 3), pady=(11, 3))

        mat_frame = ttk.Labelframe(cal_frame, text='Internal Matrix')
        mat_frame.grid(column=0, row=0, columnspan=2, sticky='NWES', padx=7, pady=7)
        self.params_labels = [[ttk.Label(mat_frame, text='?', anchor=E) for _ in range(3)] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                self.params_labels[i][j].grid(column=j, row=i, sticky=E, padx=3, pady=3)

        dist_frame = ttk.Labelframe(cal_frame, text='Lens distortion')
        dist_frame.grid(column=3, row=0, sticky='NWES', padx=7, pady=7)
        self._dist_coef_names = ['k1', 'k2', 'p1', 'p2', 'k3']
        self.dist_coef_labels = {k: ttk.Label(dist_frame, text=f'{k}: ?') for k in self._dist_coef_names}
        for row_i, k in enumerate(self._dist_coef_names):
            self.dist_coef_labels[k].grid(column=0, row=row_i, sticky='W', padx=3, pady=3)

        ttk.Button(cal_frame, text="Calibrate", command=self.calibrate).grid(column=0, row=1, padx=7, pady=5)
        ttk.Button(cal_frame, text="Save", command=self.save_calibration).grid(column=1, row=1, padx=3, pady=5)

        # Settings
        set_frame = ttk.Labelframe(self.main_frame, text='Settings')
        set_frame.grid(column=0, row=5, sticky='NWES', padx=(3, 3), pady=(11, 3))
        
        self.sqrsize_entry = self._add_setting(set_frame, 'Square size (mm):', str(self.square_size), 0)
        self.device_entry = self._add_setting(set_frame, 'Camera device:', str(self.device), 1)
        
        # Resolution row
        ttk.Label(set_frame, text='Resolution:').grid(column=0, row=2, padx=(7, 3), pady=3, sticky='E')
        self.width_entry = ttk.Entry(set_frame, width=4)
        self.width_entry.insert(0, str(self.width))
        self.width_entry.grid(column=1, row=2, sticky='W')
        ttk.Label(set_frame, text='x').grid(column=2, row=2)
        self.height_entry = ttk.Entry(set_frame, width=4)
        self.height_entry.insert(0, str(self.height))
        self.height_entry.grid(column=3, row=2, sticky='W')

        ttk.Button(set_frame, text="Update settings", command=self.update_settings).grid(column=0, row=3, columnspan=4, pady=5)

        f_status = ttk.Frame(self.main_frame, relief=RIDGE, borderwidth=1)
        f_status.grid(column=0, row=6, columnspan=2, sticky='NWES', padx=3, pady=3)
        self.status = ttk.Label(f_status, anchor=E)
        self.status.grid(column=0, row=0, sticky='NWES')

        self.thread_queue = Queue.Queue(maxsize=10)
        self.capture_thread = threading.Thread(target=self.capture_loop, kwargs={'thread_queue': self.thread_queue})
        self.capture_thread.daemon = True
        self.capture_thread.start()
        self.tk_root.after(100, self.listen_for_frame)
        self.tk_root.protocol("WM_DELETE_WINDOW", self._quit)
        self.tk_root.mainloop()

    def _add_setting(self, parent, label, default, row):
        ttk.Label(parent, text=label).grid(column=0, row=row, padx=(7, 3), pady=3, sticky='E')
        entry = ttk.Entry(parent, width=10)
        entry.insert(0, default)
        entry.grid(column=1, row=row, columnspan=3, padx=3, pady=3, sticky='W')
        return entry

    def _quit(self):
        self.tk_root.quit()
        sys.exit(0)

    def update_settings(self):
        restart_capture = False
        self.square_size = float(self.sqrsize_entry.get())
        
        w, h = int(self.width_entry.get()), int(self.height_entry.get())
        dev = self.device_entry.get()
        if dev.isdigit(): dev = int(dev)

        if self.width != w or self.height != h or self.device != dev:
            self.width, self.height, self.device = w, h, dev
            restart_capture = True

        if restart_capture:
            self.start_capture()
            self.status.configure(text='Capture restarted.')

    def start_capture(self):
        self.capture_lock.acquire()
        self.frames, self.frames_thumb, self.frames_tiny = [], [], []
        if self.capture is not None: self.capture.release()
        self.capture = cv2.VideoCapture(self.device)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        success, frame = self.capture.read()
        if success:
            self.height, self.width = frame.shape[:2]
            self.size_msg = f"Original size: {self.width} x {self.height} pixels"
        self.capture_lock.release()

    def capture_loop(self, thread_queue):
        while True:
            self.capture_lock.acquire()
            success, frame = self.capture.read()
            self.capture_lock.release()
            if success:
                frame = frame[:, :, (2, 1, 0)]
                if not thread_queue.full():
                    thread_queue.put(frame)

    def listen_for_frame(self):
        try:
            self.input_frame = self.thread_queue.get_nowait()
            tk_img = mat2tk(self.input_frame, self.thumb_size)
            self.in_frame_label.configure(image=tk_img)
            self.in_frame_label.image = tk_img
        except Queue.Empty:
            pass
        self.tk_root.after(10, self.listen_for_frame)

    def add_frame(self):
        f = self.input_frame.copy()
        self.frames.append(f)
        thumb_size = self.thumb_size
        tiny_size = (self.thumb_size[0] // 2, self.thumb_size[1] // 2)
        
        f_thumb = cv2.resize(f, thumb_size)
        f_tiny = cv2.resize(f, tiny_size)

        found, img_chess = find_chess_board(f, self.pattern_size)
        if found:
            # Resize the drawn chessboard for display
            disp_chess = cv2.resize(img_chess, thumb_size)
            self.frames_thumb.append(mat2tk(disp_chess))
        else:
            self.frames_thumb.append(mat2tk(f_thumb))

        self.frames_tiny.append(mat2tk(f_tiny))
        self.cur_frame_idx = len(self.frames) - 1
        self.update_browser()

    def prev_frame(self):
        self.cur_frame_idx = max(0, self.cur_frame_idx - 1)
        self.update_browser()

    def next_frame(self):
        self.cur_frame_idx = min(len(self.frames) - 1, self.cur_frame_idx + 1)
        self.update_browser()

    def remove_frame(self):
        if self.frames:
            self.frames.pop(self.cur_frame_idx)
            self.frames_thumb.pop(self.cur_frame_idx)
            self.frames_tiny.pop(self.cur_frame_idx)
            self.cur_frame_idx = max(0, self.cur_frame_idx - 1)
            self.update_browser()

    def update_browser(self):
        n = len(self.frames)
        if n == 0:
            f_l = f_r = self.blank_sthumb
            f_c = self.blank
            l_l = l_c = l_r = ''
        else:
            ci = self.cur_frame_idx
            li, ri = (ci - 1) % n, (ci + 1) % n
            f_l, f_r = self.frames_tiny[li], self.frames_tiny[ri]
            f_c = self.frames_thumb[ci]
            l_l, l_c, l_r = f'{li+1}/{n}', f'{ci+1}/{n}', f'{ri+1}/{n}'

        self.left_flabel.configure(image=f_l, text=l_l)
        self.center_flabel.configure(image=f_c, text=l_c)
        self.right_flabel.configure(image=f_r, text=l_r)

    def calibrate(self):
        if len(self.frames) < 3:
            tkMessageBox.showwarning('Error', 'Capture at least 3 images.')
            return

        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size

        obj_points, img_points = [], []
        for img in self.frames:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            found, corners = cv2.findChessboardCorners(gray, self.pattern_size)
            if found:
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
                cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
                img_points.append(corners)
                obj_points.append(objp)

        self.rms, self.K, self.dist_coefs, _, _ = cv2.calibrateCamera(obj_points, img_points, (self.width, self.height), None, None)

        for i in range(3):
            for j in range(3):
                self.params_labels[i][j].configure(text=f'{self.K[i, j]:.4f}')
        for k, v in zip(self._dist_coef_names, self.dist_coefs[0]):
            self.dist_coef_labels[k].configure(text=f'{k}: {v:.4f}')
        self.status.configure(text=f'RMS Error: {self.rms:.4f}')

    def save_calibration(self):
        if self.K is None: return
        filename = tkFileDialog.asksaveasfilename(defaultextension='.yaml')
        if not filename: return
        with open(filename, 'w') as f:
            f.write(f"%YAML:1.0\nCamera.fx: {self.K[0,0]}\nCamera.fy: {self.K[1,1]}\nCamera.cx: {self.K[0,2]}\nCamera.cy: {self.K[1,2]}\n")
            for i, k in enumerate(self._dist_coef_names):
                f.write(f"Camera.{k}: {self.dist_coefs[0,i]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--squaresize', type=float, default=25, help='Size of checkerboard squares in milimeters') # milimeters
    parser.add_argument('--device', default=0, help='Camera device (e.g. 0 or /dev/video0)')
    parser.add_argument('--fwidth', type=int, default=1920, help='Width of captured image')
    parser.add_argument('--fheight', type=int, default=1080, help='Height of captured image')
    parser.add_argument('--pwidth', type=int, default=8, help='Number of inner corners width-wise')
    parser.add_argument('--pheight', type=int, default=6, help='Number of inner corners height-wise')

    args = parser.parse_args()
    
    # Map device to int if it's a number
    dev = int(args.device) if args.device.isdigit() else args.device

    CalibrationApp(device=dev,
                   square_size=args.squaresize,
                   width=args.fwidth,
                   height=args.fheight,
                   pattern_size=(args.pwidth, args.pheight))