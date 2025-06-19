import base64
import io
import os
import numpy as np
import uuid
import pandas as pd
import matplotlib
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# Use a non-interactive backend for Matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Flask App Initialization ---
app = Flask(__name__)

# --- rcParams Block with Grid Settings ---
plt.rcParams.update({
    ## Text and Font Settings
    "text.usetex": False, # change to true if you have the tex in your computer
    "font.family": "serif",
    # "font.serif": ["Times New Roman"], # uncomment to try if it works for you, just to be more fancy
    "axes.labelsize": 25,
    "font.size": 16,
    "legend.fontsize": 16,
    
    ## Tick Settings
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 8, # Size of major ticks
    "ytick.major.size": 8,
    "xtick.minor.size": 4, # Size of minor ticks
    "ytick.minor.size": 4,
    
    ## Grid Settings
    "axes.grid": True, # Turn the grid on by default
    "grid.color": "lightgray", # Grid color
    "grid.linestyle": "--", # Dashed grid lines
    "grid.linewidth": 0.7, # Width of grid lines
    "grid.alpha": 0.7, # Transparency of grid lines
})

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def parse_file(filepath):
    specific_settings = {}
    header_row_number = -1
    error = None
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        dimension_lines = []
        for i in range(len(lines) - 1):
            line1, line2 = lines[i].strip().lower(), lines[i+1].strip().lower()
            if 'dimension1' in line1 and 'dimension2' in line2:
                header_row_number = i + 2
                dimension_lines = [lines[i].strip(), lines[i+1].strip()]
                break
        if header_row_number == -1:
            header_row_number = 0
            specific_settings['Parsing Info'] = 'Standard CSV assumed (no "dimension" lines found).'
        else:
            for i in range(header_row_number - 2):
                line = lines[i].strip()
                if 'Setup.Title' in line: specific_settings['Setup Title'] = line.split(',', 1)[1].strip()
                elif 'RecordTime' in line: specific_settings['Record Time'] = line.split(',', 1)[1].strip()
            if dimension_lines:
                specific_settings['Dimension1'], specific_settings['Dimension2'] = dimension_lines
    except Exception as e:
        error = f"Error reading or parsing file: {e}"
        header_row_number = -1
    return {"settings": specific_settings, "header_row": header_row_number, "error": error}

def plot_segmented_curves(df, x_col, y_col, num_curves, ax=None,
                          curves_to_plot=None, labels=None, 
                          label_source_col=None, prepend_y_col_to_label=True, 
                          file_display_name=None, num_files_plotting=1, **plot_kwargs):
    if not isinstance(df, pd.DataFrame) or df.empty or x_col not in df or y_col not in df:
        if ax is None: fig, ax = plt.subplots(); return fig, ax
        return ax.get_figure(), ax
    df_clean = df.dropna(subset=[x_col, y_col])
    total_points = len(df_clean)
    if num_curves <= 0 or total_points == 0 or total_points % num_curves != 0:
        if ax is None: fig, ax = plt.subplots(); return fig, ax
        return ax.get_figure(), ax
    points_per_curve = total_points // num_curves
    plot_indices = curves_to_plot if curves_to_plot is not None else range(num_curves)
    if any(i >= num_curves or i < 0 for i in plot_indices):
        if ax is None: fig, ax = plt.subplots(); return fig, ax
        return ax.get_figure(), ax
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.get_figure()
    
    group_label = plot_kwargs.pop('label', None)
    
    # If using per-curve labels, plot each curve individually to assign the unique label.
    if labels or label_source_col:
        for segment_idx in plot_indices:
            start, end = segment_idx * points_per_curve, (segment_idx + 1) * points_per_curve
            curve_data = df.iloc[start:end]
            
            label_for_this_curve = None
            base_label = None
            if labels and segment_idx in labels:
                base_label = labels[segment_idx]
            elif label_source_col and label_source_col in curve_data:
                unique_labels = curve_data[label_source_col].unique()
                if len(unique_labels) > 0: base_label = unique_labels[0]
            
            if base_label is not None:
                label_for_this_curve = f"{y_col}: {base_label}" if prepend_y_col_to_label else str(base_label)
            elif num_files_plotting > 1:
                 # Fallback for multi-file, per-curve labeling: use a simplified group label
                 if i == 0: label_for_this_curve = f"{y_col} ({file_display_name})"
            
            ax.plot(curve_data[x_col].values, curve_data[y_col].values, label=label_for_this_curve, **plot_kwargs)
    # Otherwise, use the much faster vectorized plotting for group labels.
    else:
        x_data = df_clean[x_col].values.reshape(num_curves, -1)
        y_data = df_clean[y_col].values.reshape(num_curves, -1)
        x_to_plot = x_data[plot_indices, :]
        y_to_plot = y_data[plot_indices, :]
        lines = ax.plot(x_to_plot.T, y_to_plot.T, **plot_kwargs)
        if group_label:
            lines[0].set_label(group_label)
            
    ax.grid(True)
    return fig, ax

@app.route('/', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        files = request.files.getlist('data_files')
        if not files or files[0].filename == '': return "No files selected.", 400
        job_id = uuid.uuid4().hex
        job_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
        os.makedirs(job_dir)
        for file in files:
            filename = secure_filename(str(file.filename))
            file.save(os.path.join(job_dir, filename))
        return redirect(url_for('job_page', job_id=job_id))
    return render_template('upload.html')

@app.route('/job/<job_id>', methods=['GET', 'POST'])
def job_page(job_id):
    job_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    if not os.path.isdir(job_dir): abort(404, "Job not found.")
    filenames = sorted([f for f in os.listdir(job_dir) if os.path.isfile(os.path.join(job_dir, f))])
    if not filenames: abort(404, "No files in this job.")

    context = {
        "job_id": job_id, "filenames": filenames, "plot_urls": [], "headers": [], 
        "settings": {}, "error": None, "form_data": request.form.to_dict(),
        "num_plots": request.args.get('num_plots', type=int)
    }

    try:
        first_file_path = os.path.join(job_dir, filenames[0])
        file_info = parse_file(first_file_path)
        if file_info["error"]: raise ValueError(file_info["error"])
        context["settings"] = file_info["settings"]
        header_row = file_info["header_row"]
        df_template = pd.read_csv(first_file_path, skiprows=header_row, low_memory=False)
        context["headers"] = df_template.columns.str.strip().tolist()

        if request.method == 'POST':
            selected_files_to_plot = request.form.getlist('files_to_plot')
            if not selected_files_to_plot:
                raise ValueError("You must select at least one file to include in the plots.")

            num_plots = int(request.form.get('num_plots', 0))
            context["num_plots"] = num_plots
            x_col = request.form.get('x_col', '').strip()
            if not x_col: raise ValueError("Global X-Column is required.")
            curves_to_plot_str = request.form.get('curves_to_plot', '').strip()
            curves_to_plot = [int(i.strip()) for i in curves_to_plot_str.split(',')] if curves_to_plot_str else None
            
            use_col_as_label = request.form.get('use_col_as_label') == 'on'
            label_source_col = request.form.get('label_source_col') if use_col_as_label else None
            curve_labels_str = request.form.get('curve_labels', '').strip()
            curve_labels = [s.strip() for s in curve_labels_str.split(',')] if curve_labels_str else []
            labels_dict = {}
            if not label_source_col:
                if curves_to_plot and len(curves_to_plot) == len(curve_labels):
                    labels_dict = dict(zip(curves_to_plot, curve_labels))
                elif not curves_to_plot and curve_labels:
                    labels_dict = {i: label for i, label in enumerate(curve_labels)}
            prepend_y_col = request.form.get('prepend_y_col') == 'on'

            for i in range(1, num_plots + 1):
                numerator = request.form.get(f'numerator_{i}')
                denominator = request.form.get(f'denominator_{i}')
                y_columns_details = []
                for j in range(1, 6):
                    y_col = request.form.get(f'y_col_{i}_{j}')
                    if y_col:
                        style = request.form.get(f'linestyle_{i}_{j}', 'solid')
                        y_columns_details.append({"name": y_col, "style": style})

                if not y_columns_details and not (numerator and denominator): continue
                fig, ax = plt.subplots(figsize=(12, 8))
                final_ylabel = ""
                
                plot_paths = []
                if numerator and denominator:
                    final_ylabel = request.form.get(f'ylabel_{i}', '').strip() or f"({numerator} / {denominator})"
                    plot_paths = [{"y_col": f"({numerator} / {denominator})", "numerator": numerator, "denominator": denominator, "style": "solid"}]
                elif y_columns_details:
                    final_ylabel = request.form.get(f'ylabel_{i}', '').strip() or ", ".join(d['name'] for d in y_columns_details)
                    plot_paths = [{"y_col": d['name'], "numerator": None, "style": d['style']} for d in y_columns_details]

                for path in plot_paths:
                    for file_idx, filename in enumerate(filenames):
                        if filename not in selected_files_to_plot: continue
                        
                        num_curves = request.form.get(f'num_curves_{file_idx}', type=int)
                        if not num_curves: raise ValueError(f"'Total # of Curves' is required for {filename}.")
                        df = pd.read_csv(os.path.join(job_dir, filename), skiprows=header_row, low_memory=False)
                        df.columns = df.columns.str.strip()
                        label_col_data = df[label_source_col].copy() if label_source_col and label_source_col in df else None
                        
                        y_col_to_plot = path["y_col"]
                        if path.get("numerator"):
                            num, den = path["numerator"], path["denominator"]
                            if num not in df.columns or den not in df.columns: raise ValueError(f"'{num}' or '{den}' not in {filename}")
                            df[y_col_to_plot] = pd.to_numeric(df[num], errors='coerce') / pd.to_numeric(df[den], errors='coerce')
                        
                        df_numeric = df.apply(pd.to_numeric, errors='coerce')
                        if label_col_data is not None: df_numeric[label_source_col] = label_col_data
                        
                        if scale_col := request.form.get(f'scale_col_{i}', '').strip():
                            if scale_factor_str := request.form.get(f'scale_factor_{i}'):
                                if scale_col in df_numeric.columns:
                                    df_numeric[scale_col] *= float(scale_factor_str)
                        
                        file_alias = request.form.get(f'file_alias_{file_idx}', '').strip() or os.path.splitext(filename)[0]
                        group_label = f"{y_col_to_plot} ({file_alias})"
                        
                        plot_segmented_curves(
                            df=df_numeric, x_col=x_col, y_col=y_col_to_plot, num_curves=num_curves, ax=ax,
                            curves_to_plot=curves_to_plot, labels=labels_dict, label_source_col=label_source_col, 
                            prepend_y_col_to_label=prepend_y_col, file_display_name=file_alias,
                            num_files_plotting=len(selected_files_to_plot), lw=2, linestyle=path['style']
                        )
                
                final_xlabel = str(request.form.get('xlabel', '') or x_col)
                final_ylabel = str(final_ylabel)
                auto_title = f'Plot of {final_ylabel} vs {final_xlabel}'
                final_title = str(request.form.get(f'title_{i}', '').strip() or auto_title)
                ax.set_title(final_title); ax.set_xlabel(final_xlabel); ax.set_ylabel(final_ylabel)
                if request.form.get(f'legend_{i}') == 'on': ax.legend()
                if request.form.get('x_log') == 'on': ax.set_xscale('log')
                if request.form.get(f'y_log_{i}') == 'on': ax.set_yscale('log')
                def to_float(val): return float(val) if val else None
                ax.set_xlim(left=to_float(request.form.get('x_min')), right=to_float(request.form.get('x_max')))
                ax.set_ylim(bottom=to_float(request.form.get(f'y_min_{i}')), top=to_float(request.form.get(f'y_max_{i}')))
                
                # --- CHANGE IS HERE: Added dpi=300 ---
                img = io.BytesIO()
                fig.savefig(img, format='png', bbox_inches='tight', dpi=300)
                img.seek(0)
                context["plot_urls"].append(base64.b64encode(img.getvalue()).decode('utf8'))
                plt.close(fig)
    except Exception as e:
        context["error"] = f"An error occurred: {e}"

    return render_template('plot.html', **context)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
