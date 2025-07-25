
"""
    @file:              utils.py
    @Author:            Moustafa Amine Bezzahi, Ihssene Brahimi 

    @Creation Date:     07/2024
    @Last modification: 08/2024

    @Description:       This file is used to define the UI application that uses the pipeline.
"""
from utils import *
from evaluation.classification.evaluate import *
from evaluation.segmentation.predicts import *
from data.transforms import get_bbox_transforms, get_srm_transforms, get_kidney_transforms
#########################
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import SimpleITK as sitk
import os
import matplotlib.patches as patches
from scipy.ndimage import label, find_objects
import monai.transforms as mt
from typing import Tuple, Optional
from PIL import Image, ImageTk
import webbrowser


# Model_init
def kidney_segmentor(input_vol):
    num_classes = 3
    # model_folder = "/projects/renal/Notebooks/Models/semisup/model"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = monai.networks.nets.BasicUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            features=(16, 32, 64, 128, 256, 32),
            dropout=0.1,
        )
    # model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join("/projects/renal/Notebooks/Models/semisup/model/best_metric_model_full_256.pth")))
    model.eval()
    seg = model(input_vol)
    return seg

def lesion_segmentor(input_vol):
    num_classes = 2
    features = (16, 32, 64, 128, 32, 16)
    student_dropout = 0.1 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = monai.networks.nets.BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        features=features,  
        dropout=student_dropout,
    )

    model.load_state_dict(torch.load("/projects/renal/srm-detection-main/models/weights/segmentation/srm/best_fully_supervised_srm_model_fold_5.pth"))

    model.to(device)
    model.eval()

    seg = model(input_vol)
    return seg

# Ignore this
ct_seg_path =""
ct_seg1 = ''
############
original_vol = np.random.rand(100, 256, 256)
kidney_vol = np.random.rand(100, 256, 256)
tumor_vol = np.random.rand(100, 256, 256)
bboxes = []
kidney_seg = np.random.rand(100, 256, 256)
tumor_seg = np.random.rand(100, 256, 256)
ccRCC_pred = ""
ccRCC_prob = ""
grade_pred = ""
grade_prob= ""
pap_pred= ""
pap_prob= ""
################# pipline Fun

####### Funs

def browse_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        file_entry.delete(0, tk.END)
        file_entry.insert(0, file_path)
        
def save_files():
    
    save_dir = filedialog.askdirectory()

    if not save_dir:
        return  # Exit if no directory is selected
    
    try:
        file_save.delete(0, tk.END)
        file_save.insert(0, save_dir)
        # Save volumes
        sitk.WriteImage(sitk.GetImageFromArray(original_vol), os.path.join(save_dir, 'Transformed_ct_vol.nrrd'))
        sitk.WriteImage(sitk.GetImageFromArray(kidney_seg), os.path.join(save_dir, 'Kidney_segmented.seg.nrrd'))
        sitk.WriteImage(sitk.GetImageFromArray(tumor_seg), os.path.join(save_dir, 'Lesion_segmented.seg.nrrd'))
        
        
        classification_data = [
            ("ccRCC vs non-ccRcc", ccRCC_pred, f"{ccRCC_prob}%"),
            ("ccRCC vs Pap", pap_pred, f"{pap_prob}%"),
            ("Grade", grade_pred, f"{grade_prob}%")
        ]
        bounding_boxes = bounding_box(kidney_seg)
        
        with open(os.path.join(save_dir, 'classification_results.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Pathology', 'Predictions', 'Probabilities'])
            writer.writerows(classification_data)
            
            writer.writerow([])
            writer.writerow(['Bounding Boxes'])
            writer.writerow(['z_start', 'y_start', 'x_start', 'z_length', 'y_length', 'x_length'])
            writer.writerows(bounding_boxes)
            
            notes = note_entry.get("1.0", tk.END).strip()
            writer.writerow([]) 
            writer.writerow(['Medical Notes'])
            writer.writerow([notes])
            
        messagebox.showinfo("Save Results", f"Results saved successfully to {save_dir}")
    except Exception as e:
        
        messagebox.showerror("Error", f"Failed to save results: {e}")
        
    print('saved')
def add_notes():
    notes = note_entry.get("1.0", tk.END).strip()
    messagebox.showinfo("Add Notes", "Notes added successfully")
    
def start():
    
    file_path = file_entry.get()
    if file_path:
        ct_vol = sitk.ReadImage(file_path)
        display_ct_info(ct_vol, file_path)
        
        print("Starting Predictions for Selected file:")
        print(f"{file_path}")
    
def display_ct_info(ct_vol, file_path):

    global original_vol, kidney_vol, tumor_vol, boxes, kidney_seg, tumor_seg, ccRCC_pred, ccRCC_prob, grade_pred, grade_prob, pap_pred, pap_prob

    file_name = os.path.basename(file_path)
    
    dim = ct_vol.GetSize()
    spax = ct_vol.GetSpacing()
    org = ct_vol.GetOrigin()
    ort = ct_vol.GetDirection()
    # print(ort)
    min_intensity = (sitk.GetArrayFromImage(ct_vol)).min()
    max_intensity = (sitk.GetArrayFromImage(ct_vol)).max()
    # Translate the direction matrix to letters
    x_orientation = "R" if ort[0] > 0 else "L"
    y_orientation = "A" if ort[4] > 0 else "P"
    z_orientation = "S" if ort[8] > 0 else "I"
    orientation_str = f"{x_orientation} {y_orientation} {z_orientation}"
    
    
    props_table.set(props_table.get_children()[0], column="Value", value=f"{file_name}")
    props_table.set(props_table.get_children()[1], column="Value", value=f"{dim}")
    props_table.set(props_table.get_children()[2], column="Value", value=f"{spax}")
    props_table.set(props_table.get_children()[3], column="Value", value=f"{org}", )
    props_table.set(props_table.get_children()[4], column="Value", value=orientation_str)
    props_table.set(props_table.get_children()[5], column="Value", value=f"{min_intensity:.2f} to {max_intensity:.2f}")
    # ct_vol.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    original_vol = sitk.GetArrayFromImage(ct_vol)
    kidney_trans = get_kidney_transforms(ct_vol, mode ='unlabeled_train', keys='img')
    kidney_vol = kidney_trans(file_path)
    kidney_seg = kidney_segmentor(kidney_vol)
    kidney_seg = (kidney_seg == 0) #getting the predicted ROI segmentation

    boxes = bounding_box(kidney_seg)
    
    tumor_trans = get_srm_transforms(mode="unlabeled_train", keys="img")
    
    tumor_vol = tumor_trans(file_path)
    tumor_seg = lesion_segmentor(tumor_vol)

    z = kidney_vol.shape[0] 
    z1 = tumor_vol.shape[0] 
    
    switch.config(command=lambda: 
                  update_kidney_plot(slice_index=scale1.get(), 
                              ct=kidney_vol, seg=kidney_seg, ax=ax1, 
                              canvas=canvas1, slice_label=slice_label1, 
                              show=show_seg.get(), view=view_combobox.get()))
        
    scale1.config(to=z-1, command=lambda value: 
                  update_kidney_plot(slice_index=value, ct=kidney_vol, 
                              seg=kidney_seg, ax=ax1, canvas=canvas1, 
                              slice_label=slice_label1, 
                              show=show_seg.get(), view=view_combobox.get()))
    
    
    switch1.config(command=lambda: (update_classif(corr.get(),classif_tree), update_lesion_plot(slice_index=scale3.get(), 
                ct=tumor_vol, seg=tumor_seg, ax=ax3, 
                canvas=canvas3, slice_label=slice_label3, bounding_boxes=boxes, 
                show=sh_seg.get(), view=view_combobox2.get(),
                corrected_boxes=corr.get()), update_roi_plot(slice_index=scale2.get(), 
                                                                 ct=kidney_vol,  
                                                                 ax=ax2, 
                                                                 canvas=canvas2, 
                                                                 slice_label=slice_label2, 
                                                                 bounding_boxes=boxes, 
                                                                 sw=corr.get())))
    scale2.config(to=z-1, command=lambda value: update_roi_plot(slice_index=value, ct=kidney_vol,  ax=ax2, canvas=canvas2, slice_label=slice_label2, bounding_boxes=boxes, sw=corr.get()))
    
          
    switch2.config(command=lambda: 
                  update_lesion_plot(slice_index=scale3.get(), 
                              ct=tumor_vol, seg=tumor_seg, ax=ax3, 
                              canvas=canvas3, slice_label=slice_label3, bounding_boxes=boxes, 
                              show=sh_seg.get(), view=view_combobox2.get(),
                              corrected_boxes=corr.get()))
        
    scale3.config(to=z1-1, command=lambda value: 
                  update_lesion_plot(slice_index=value, ct=tumor_vol, 
                              seg=tumor_seg, ax=ax3, canvas=canvas3, 
                              slice_label=slice_label3, bounding_boxes=boxes,
                              show=sh_seg.get(), view=view_combobox2.get(),
                              corrected_boxes=corr.get()))
    

   
def update_scales():
    selected_view = view_combobox.get()
    selected_view2 = view_combobox2.get()
    sw = corr.get()
    
    if selected_view == 'Axial':
        z1 = kidney_vol.shape[0]
    elif selected_view == 'Coronal':
        z1 = kidney_vol.shape[1]
    elif selected_view == 'Sagittal':
        z1 = kidney_vol.shape[2]

    # if sw :
    #     if selected_view2 == 'Axial':
    #         z3 = boxes[0][3] 
    #     elif selected_view2 == 'Coronal':
    #         z3 = boxes[0][4]  
    #     elif selected_view2 == 'Sagittal':
    #         z3 = boxes[0][5]  
    # else :
    #     if selected_view2 == 'Axial':
    #         z3 = boxes[1][3] 
    #     elif selected_view2 == 'Coronal':
    #         z3 = boxes[1][4]  
    #     elif selected_view2 == 'Sagittal':
    #         z3 = boxes[1][5]
        
    if selected_view2 == 'Axial':
        z3 = tumor_vol.shape[0]
    elif selected_view2 == 'Coronal':
        z3 = tumor_vol.shape[1]
    elif selected_view2 == 'Sagittal':
        z3 = tumor_vol.shape[2]
    
    # Update the scales
    scale1.config(to=z1-1)
    update_kidney_plot(slice_index=0, ct=kidney_vol, 
                seg=kidney_seg, ax=ax1, canvas=canvas1, 
                slice_label=slice_label2, 
                show=show_seg.get(), view=selected_view)
    
    scale3.config(to=z3-1)
    update_lesion_plot(slice_index=0, ct=tumor_vol, 
                seg=tumor_seg, ax=ax3, canvas=canvas3, 
                slice_label=slice_label3, bounding_boxes=boxes,
                show=sh_seg.get(), view=selected_view2, corrected_boxes=sw)

def update_kidney_plot(
    slice_index, ct, seg, ax, canvas, slice_label, view='Axial', show=False):

    slice_index = int(round(float(slice_index)))
    ax.clear()
    
    # Adjust slice and view
    if view == 'Axial':
        ct_slice = ct[slice_index, :, :]
        seg_slice = seg[slice_index, :, :] if seg is not None else None
    elif view == 'Coronal':
        ct_slice = ct[:, slice_index, :]
        seg_slice = seg[:, slice_index, :] if seg is not None else None
    elif view == 'Sagittal':
        ct_slice = ct[:, :, slice_index]
        seg_slice = seg[:, :, slice_index] if seg is not None else None
        
    # Display the CT slice
    ax.imshow(ct_slice, cmap='gray')

    # Show segmentation 
    if (show):
        if seg_slice is not None:
            ax.imshow(seg_slice, cmap='jet', alpha=0.3)         
    ax.set_title(f'Slice {slice_index} ({view} View)')
    canvas.draw()
    
    slice_label.config(text=f'Slice: {slice_index} ({view} View)')

    
def update_roi_plot(
    slice_index, ct, ax, canvas, slice_label,
    bounding_boxes: Optional[Tuple[Tuple[int, int, int, int, int, int], Tuple[int, int, int, int, int, int]]] = None,
    view='Axial', sw=False):
    
    slice_index = int(round(float(slice_index)))
    ax.clear()
    
    # Adjust slice and view
    if view == 'Axial':
        ct_slice = ct[slice_index, :, :]
        # seg_slice = seg[slice_index, :, :] if seg is not None else None
    elif view == 'Coronal':
        ct_slice = ct[:, slice_index, :]
        # seg_slice = seg[:, slice_index, :] if seg is not None else None
    elif view == 'Sagittal':
        ct_slice = ct[:, :, slice_index]
        # seg_slice = seg[:, :, slice_index] if seg is not None else None
    
    
    # Define colors for bounding boxes
    colors = ['r', 'g']
    
    # bounding boxes
    if bounding_boxes and slice_label ==slice_label2 :
        for i, box in enumerate(bounding_boxes):
            if sw:
                colors = ['g', 'r']
               
            z_start, y_start, x_start, z_length, y_length, x_length = box

            # Check which view and adjust box display
            if view == 'Axial' and z_start <= slice_index < z_start + z_length:
                x, y, width, height = x_start, y_start, x_length, y_length
            elif view == 'Coronal' and y_start <= slice_index < y_start + y_length:
                x, y, width, height = x_start, z_start, x_length, z_length
            elif view == 'Sagittal' and x_start <= slice_index < x_start + x_length:
                x, y, width, height = y_start, z_start, y_length, z_length
            else:
                continue

            color = colors[i % len(colors)]
            rect = patches.Rectangle((x, y), width, height, edgecolor=color, facecolor='none', lw=2)
            ax.add_patch(rect)
    # Display the CT slice
    ax.imshow(ct_slice, cmap='gray')
    ax.set_title(f'Slice {slice_index} ({view} View)')
    canvas.draw()
    
    slice_label.config(text=f'Slice: {slice_index}')



def update_lesion_plot(
    slice_index, ct, seg, ax, canvas, slice_label,
    bounding_boxes: Optional[Tuple[Tuple[int, int, int, int, int, int], Tuple[int, int, int, int, int, int]]] = None,
    view='Axial', show=False, corrected_boxes=False):
   
    slice_index = int(round(float(slice_index)))
    ax.clear()
    seg = (seg == 1) if corrected_boxes else (seg == 2) 
    # Adjust slice and view
    if view == 'Axial':
        ct_slice = ct[slice_index, :, :]
        seg_slice = seg[slice_index, :, :] if seg is not None else None
    elif view == 'Coronal':
        ct_slice = ct[:, slice_index, :]
        seg_slice = seg[:, slice_index, :] if seg is not None else None
    elif view == 'Sagittal':
        ct_slice = ct[:, :, slice_index]
        seg_slice = seg[:, :, slice_index] if seg is not None else None
            
    if (bounding_boxes is not None) : 
        if corrected_boxes :
            box = bounding_boxes[1]  
        else :
            box = bounding_boxes[0]
 
    
        z_start, y_start, x_start, z_length, y_length, x_length = box
    
        if view == 'Axial' and z_start <= slice_index < z_start + z_length:
            # Crop CT and segmentation slices to the bounding box
            ct_slice = ct_slice[y_start:y_start + y_length, x_start:x_start + x_length]
            if seg_slice is not None:
                seg_slice = seg_slice[y_start:y_start + y_length, x_start:x_start + x_length]
        elif view == 'Coronal' and y_start <= slice_index < y_start + y_length:
            ct_slice = ct_slice[z_start:z_start + z_length, x_start:x_start + x_length]
            if seg_slice is not None:
                seg_slice = seg_slice[z_start:z_start + z_length, x_start:x_start + x_length]
        elif view == 'Sagittal' and x_start <= slice_index < x_start + x_length:
            ct_slice = ct_slice[z_start:z_start + z_length, y_start:y_start + y_length]
            if seg_slice is not None:
                seg_slice = seg_slice[z_start:z_start + z_length, y_start:y_start + y_length]
    
        # Display cropped CT slice
        ax.imshow(ct_slice, cmap='gray')
        if show and seg_slice is not None:
            ax.imshow(seg_slice, cmap='jet', alpha=0.3)
      
    ax.set_title(f'Slice {slice_index} ({view} View)')
    canvas.draw()
    
    slice_label.config(text=f'Slice: {slice_index} ({view} View)')   

def update_classif(sw, classif_tree):
    global ccRCC_pred, ccRCC_prob, grade_pred, grade_prob, pap_pred, pap_prob

    if sw :
        # get 1st bbox classif_predictions
        ccRCC_pred='non-ccRCC' 
        ccRCC_prob= '0.83 ± 0.09'
        grade_pred= 'Low'
        grade_prob= '0.69 ± 0.3'
        pap_pred= 'papRCC'
        pap_prob='0.79 ± 0.14'
    
    else :
        # get 2st bbox classif_predictions
        ccRCC_pred='ccRCC'
        ccRCC_prob= '0.8 ± 0.16'
        grade_pred= 'High'
        grade_prob= '0.75 ± 0.21'
        pap_pred= 'ccRCC'
        pap_prob='0.95 ± 0.04'
        
    classif_tree.set(classif_tree.get_children()[0], column="Column 2", value=f"{ccRCC_pred}")
    classif_tree.set(classif_tree.get_children()[1], column="Column 2", value=f"{grade_pred}")
    classif_tree.set(classif_tree.get_children()[2], column="Column 2", value=f"{pap_pred}")
    classif_tree.set(classif_tree.get_children()[0], column="Column 3", value=f"{ccRCC_prob}")
    classif_tree.set(classif_tree.get_children()[1], column="Column 3", value=f"{grade_prob}")
    classif_tree.set(classif_tree.get_children()[2], column="Column 3", value=f"{pap_prob}")

      
#######################
def reorder_bounding_boxes(bounding_boxes: Optional[Tuple[Tuple[int, int, int, int, int, int], Tuple[int, int, int, int, int, int]]]):
    if bounding_boxes is None:
        return None
    
    box1, box2 = bounding_boxes
    reordered_bounding_boxes = (box2, box1)
    
    return reordered_bounding_boxes        
#######################
def bounding_box(mask):
    labeled, num_features = label(mask)  # Label connected components
    objects = find_objects(labeled) 
    
    boxes = []
    for obj in objects:
        x_start, x_end = obj[0].start, obj[0].stop
        y_start, y_end = obj[1].start, obj[1].stop
        z_start, z_end = obj[2].start, obj[2].stop
        x_len = x_end - x_start
        y_len = y_end - y_start
        z_len = z_end - z_start
        volume = x_len * y_len * z_len
        boxes.append((x_start, y_start, z_start, x_len, y_len, z_len, volume))
    
    # Sort boxes by volume in descending order and take the two largest
    boxes = sorted(boxes, key=lambda x: x[6], reverse=True)[:2]
    
    
    boxes = [(x_start, y_start, z_start, x_len, y_len, z_len) for x_start, y_start, z_start, x_len, y_len, z_len, _ in boxes]
    
    return boxes
def open_documentation():
    url = "" #link to doc
    webbrowser.open(url)

def open_video_guide():
    url = "" #link to video_tuto 
    webbrowser.open(url)
def switch_theme():
    if d.get() :
        style.theme_use("forest-light")
        d.set(value=True)
        mode_button.config(text='Switch to Dark Mode')
    else : 
        style.theme_use("forest-dark")
        d.set(value=False)
        mode_button.config(text='Switch to Light Mode')
        

############################# Main()

root = tk.Tk()
root.title("SRM autoDetect")
root.iconbitmap('logo.ico')
root.option_add("*tearOff", False)  # Always a good idea

menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

file_menu = tk.Menu(menu_bar)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Open", command=browse_file)
file_menu.add_command(label="Save", command=save_files)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

help_menu = tk.Menu(menu_bar)
menu_bar.add_cascade(label="Help", menu=help_menu)
help_menu.add_command(label="Documentation", command=open_documentation)
help_menu.add_command(label="Video Guide", command=open_video_guide)
help_menu.add_command(label="Dark Mode", command=switch_theme)


# Make the app responsive
root.columnconfigure(index=0, weight=1)
root.columnconfigure(index=1, weight=1)
root.columnconfigure(index=2, weight=1)
root.rowconfigure(index=0, weight=1)
root.rowconfigure(index=1, weight=1)
root.rowconfigure(index=2, weight=1)


style = ttk.Style(root)


root.tk.call("source", "forest-dark.tcl")
style.theme_use("forest-dark")
d = tk.BooleanVar(value=False)


#### Controllers
g = tk.DoubleVar(value=75.0)
show_seg = tk.BooleanVar(value=False)  
corr = tk.BooleanVar(value=False)  
sh_seg =  tk.BooleanVar(value=False)

check_frame = ttk.LabelFrame(root, text="Welcome to SRMDetector" ,padding=(20, 10))
check_frame.grid(row=0, column=0, padx=5, pady=15, sticky="nsew")

check_frame.columnconfigure(0, weight=2)
check_frame.columnconfigure(1, weight=3)
check_frame.rowconfigure(0, weight=2)
check_frame.rowconfigure(1, weight=1)
check_frame.rowconfigure(2, weight=1)

desc_label = tk.Label(
    check_frame,border=40,
    text="The SRMDetector Desktop Application is a specialized tool designed to assist medical professionals in the analysis and interpretation of Computed Tomography (CECT) imaging data for small renal masses",
    font=("Times New Roman", 13),
    wraplength=300,  
    justify='center')
desc_label.grid(row=0, column=0, columnspan=2, ipadx=10, pady=10, sticky="nsew")

file_entry = ttk.Entry(check_frame, width=40)
file_entry.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

file_entry.insert(0, 'Insert Volume Path ')
file_entry.bind("<FocusIn>", lambda e: file_entry.delete(0, tk.END))


browse_button = ttk.Button(check_frame, text=" Browse ", command=browse_file, width=10)
browse_button.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

start_button = ttk.Button(check_frame, text=" Start ", default ="active",width=16, command=start)
start_button.grid(row=2, column=0, padx=5, pady=5 ,sticky="nsew")

# img = Image.open('helpo.png')
# img_tk = ImageTk.PhotoImage(img)
# help_img = tk.Label(check_frame, image=img_tk, height=2, width=2)
# help_img.grid(row=2, column=2 ,padx=(45,5), pady=5)

mode_button = ttk.Checkbutton(check_frame, text="Light Mode", style='Switch', variable=d, command=switch_theme)
mode_button.grid(row=2, column=1, padx=25, pady=5 ,sticky="nsew")

separator = ttk.Separator(root)
separator.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

# props_frame
props_frame = ttk.LabelFrame(root, text="Volume Properties", padding=(5, 5))
props_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
props_frame.columnconfigure(0, weight=2)  
props_frame.rowconfigure(0, weight=1)

props_table = ttk.Treeview(props_frame, columns=("Property", "Value"), show="headings")
props_table.grid(row=0, column=0, padx=0, pady=10 ,columnspan=2, sticky="n")


props_table.column("Property", width=100, anchor="w")
props_table.column("Value", width=150 ,anchor="w")
props_table.heading("Property", text="Property")
props_table.heading("Value", text="Value")


# Insert initial properties
props_table.insert("", "end", values=("File Name", ""))
props_table.insert("", "end", values=("Dimensions", ""))
props_table.insert("", "end", values=("Spacing", ""))
props_table.insert("", "end", values=("Origin", ""))
props_table.insert("", "end", values=("Orientation", ""))
props_table.insert("", "end", values=("Intensity Range", ""))

widgets_frame = ttk.Frame(root, padding=(0, 0, 0, 10))
widgets_frame.grid(row=0, column=1, padx=10, pady=(30, 10), sticky="nsew", rowspan=3)
widgets_frame.columnconfigure(index=0, weight=1)
paned = ttk.PanedWindow(root)
paned.grid(row=0, column=2, pady=(25, 5), sticky="nsew", rowspan=3)

paned = ttk.PanedWindow(root)
paned.grid(row=0, column=2, pady=(25, 5), sticky="nsew", rowspan=3)

pane_1 = ttk.Frame(paned)
paned.add(pane_1, weight=3)

notebook = ttk.Notebook(pane_1)


tab_1 = ttk.Frame(notebook)
tab_1.columnconfigure(index=0, weight=1)
tab_1.columnconfigure(index=1, weight=1)
tab_1.rowconfigure(index=0, weight=1)
tab_1.rowconfigure(index=1, weight=1)

notebook.add(tab_1, text="Kidney Detection")

fig1, ax1 = plt.subplots(figsize=(10, 8))
canvas1 = FigureCanvasTkAgg(fig1, master=tab_1)
canvas1.get_tk_widget().grid(row=0, column=0, padx=10, pady=10,columnspan=2)

slice_label1 = ttk.Label(tab_1, text="Slice: 0", justify="center")
slice_label1.grid(row=2, column=2, padx=(10, 10), pady=(10, 10))  

g1 = tk.DoubleVar()
scale1 = ttk.Scale(tab_1, from_=0, to=99, orient='horizontal', variable=g1)
scale1.grid(row=2, column=1, padx=(10, 10), pady=(10, 10), sticky="ew") 

switch = ttk.Checkbutton(tab_1, text="Show Kidneys Segmentation", variable=show_seg)
switch.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")

view_combobox = ttk.Combobox(tab_1, width=10, values=["Axial", "Coronal", "Sagittal"], state='readonly')
view_combobox.grid(row=0, column=2, padx=(50, 50), pady=(10, 10))
view_combobox.set("Axial")
view_combobox.bind("<<ComboboxSelected>>", lambda event: update_scales())
update_kidney_plot(0, original_vol, kidney_vol, ax1, canvas1, slice_label1)


tab_2 = ttk.Frame(notebook)

tab_2.columnconfigure(index=0, weight=1)
tab_2.columnconfigure(index=1, weight=1)
tab_2.rowconfigure(index=0, weight=1)
tab_2.rowconfigure(index=1, weight=1)
notebook.add(tab_2, text="ROI Detection")


fig2, ax2 = plt.subplots(figsize=(10, 8))
canvas2 = FigureCanvasTkAgg(fig2, master=tab_2)
canvas2.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, columnspan=2)

slice_label2 = ttk.Label(tab_2, text="Slice: 0", justify="center")
slice_label2.grid(row=2, column=2, padx=(5, 5), pady=(10, 10))  

g2 = tk.DoubleVar()
scale2 = ttk.Scale(tab_2, from_=0, to=99, orient='horizontal', variable=g2)
scale2.grid(row=2, column=1, padx=(10, 10), pady=(10, 10), sticky="ew") 

switch1 = ttk.Checkbutton(tab_2, text="Correct Bounding Box", style='Switch' ,variable=corr)
switch1.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

update_roi_plot(0, original_vol, ax2, canvas2, slice_label2)



tab_3 = ttk.Frame(notebook)

tab_3.columnconfigure(index=0, weight=1)
tab_3.columnconfigure(index=1, weight=1)
tab_3.rowconfigure(index=0, weight=1)
tab_3.rowconfigure(index=1, weight=1)
notebook.add(tab_3, text="Lesion Detection")


fig3, ax3 = plt.subplots(figsize=(10, 8))
canvas3 = FigureCanvasTkAgg(fig3, master=tab_3)
canvas3.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, columnspan=2)

slice_label3 = ttk.Label(tab_3, text="Slice: 0", justify="center")
slice_label3.grid(row=2, column=2, padx=(10, 10), pady=(10, 10)) 

g3 = tk.DoubleVar()
scale3 = ttk.Scale(tab_3, from_=0, to=99, orient='horizontal', variable=g3)
scale3.grid(row=2, column=1, padx=(10, 10), pady=(10, 10), sticky="ew") 

switch2 = ttk.Checkbutton(tab_3, text="Show Lesion Segmentation", variable=sh_seg)
switch2.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

view_combobox2 = ttk.Combobox(tab_3, width=10, values=["Axial", "Coronal", "Sagittal"], state='readonly')
view_combobox2.grid(row=0, column=2, padx=(50, 50), pady=(10, 10))
view_combobox2.set("Axial")
view_combobox2.bind("<<ComboboxSelected>>", lambda event: update_scales())
update_lesion_plot(0, original_vol, kidney_vol , ax3, canvas3, slice_label3)



tab_4 = ttk.Frame(notebook)

notebook.add(tab_4, text="Classifications")

tab_4.columnconfigure(index=0, weight=1)
tab_4.columnconfigure(index=1, weight=1)
tab_4.rowconfigure(index=0, weight=1)
tab_4.rowconfigure(index=1, weight=1)
tab_4.rowconfigure(index=2, weight=1)
tab_4.rowconfigure(index=3, weight=1)


classif_tree = ttk.Treeview(
    tab_4, 
    selectmode="extended", 
    columns=("Column 1", "Column 2", "Column 3"), 
    show='headings'
)
classif_tree.heading("Column 1", text="Pathology", anchor="w")
classif_tree.heading("Column 2", text="Prediction", anchor="w")
classif_tree.heading("Column 3", text="Probability", anchor="w")


initial_values1 = ["ccRCC vs non-ccRcc", "Grade", "ccRCC vs Pap"]
initial_values2 = ["", "", ""]  
initial_values3 = ["", "", ""] 


for i in range(3):
    classif_tree.insert("", "end", values=(initial_values1[i], initial_values2[i], initial_values3[i]))


classif_tree.configure(height=5)
classif_tree.grid(row=0, column=0, columnspan=2, padx=(10, 10), pady=(10, 10))



notes_label = ttk.Button(tab_4, text="Add Notes", command=add_notes)
notes_label.grid(row=1, column=0, padx=(5, 0), pady=(5, 5), sticky="e")


note_entry = tk.Text(tab_4, font=("Georgian", 12), width=70, height=10)
note_entry.grid(row=1, column=1, padx=(5, 100), pady=(5, 5), sticky="e")


file_save = ttk.Entry(tab_4, width=80)
file_save.grid(row=2, column=1, padx=(0, 40), pady=(5, 5))


save_button = ttk.Button(tab_4, text="Save results", command=lambda: print("Save clicked"))
save_button.grid(row=2, column=0, padx=(5, 5), pady=(5, 5), sticky="e")


notebook.pack(expand=True, fill="both", padx=5, pady=5)

# Sizegrip
sizegrip = ttk.Sizegrip(root)
sizegrip.grid(row=100, column=100, padx=(0, 5), pady=(0, 5))

# Center the window, and set minsize
root.update()
root.minsize(root.winfo_width(), root.winfo_height())
x_cordinate = int((root.winfo_screenwidth()/2) - (root.winfo_width()/2))
y_cordinate = int((root.winfo_screenheight()/2) - (root.winfo_height()/2))
root.geometry("+{}+{}".format(x_cordinate, y_cordinate))


root.mainloop()


