import base64
import io
import os
import uuid
import pandas as pd
import numpy as np
import matplotlib
from flask import Flask, render_template, request, redirect, url_for, abort, session
from werkzeug.utils import secure_filename
import re
import traceback

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "axes.labelsize": 25,
    "font.size": 16,
    "legend.fontsize": 16,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "xtick.minor.size": 4,
    "ytick.minor.size": 4,
    "axes.grid": True,
    "grid.color": "lightgray",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.7,
})

app = Flask(__name__)
app.secret_key = 'a_final_secret_key_for_all_features_and_stability'

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
        for i in range(len(lines) - 1):
            line1_no_space = lines[i].strip().lower().replace(" ", "")
            line2_no_space = lines[i+1].strip().lower().replace(" ", "")
            if 'dimension1' in line1_no_space and 'dimension2' in line2_no_space:
                header_row_number = i + 2
                break
        if header_row_number == -1:
            for i, line in enumerate(lines):
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith('#'):
                    header_row_number = i
                    break
        if header_row_number == -1: raise ValueError("Could not find a valid header row.")
        for i in range(header_row_number):
            line = lines[i].strip()
            if not line: continue
            if re.search(r'Setup\.?Title', line, re.IGNORECASE):
                specific_settings['Setup Title'] = line.split(',', 1)[1].strip() if len(line.split(',', 1)) > 1 else ''
            elif 'RecordTime' in line:
                specific_settings['Record Time'] = line.split(',', 1)[1].strip() if len(line.split(',', 1)) > 1 else ''
            elif 'dimension' in line.lower():
                if 'dimension 1' in line.lower().replace(" ",""): specific_settings['Dimension 1'] = line
                if 'dimension 2' in line.lower().replace(" ",""): specific_settings['Dimension 2'] = line
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
    if ax is None: fig, ax = plt.subplots(figsize=(12, 8));
    else: fig = ax.get_figure()
    group_label = plot_kwargs.pop('label', None)
    for i, segment_idx in enumerate(plot_indices):
        start, end = segment_idx * points_per_curve, (segment_idx + 1) * points_per_curve
        curve_data = df_clean.iloc[start:end]
        label_for_this_curve = None
        base_label = None
        if labels and segment_idx in labels: base_label = labels[segment_idx]
        elif label_source_col and label_source_col in curve_data:
            unique_labels = curve_data[label_source_col].dropna().unique()
            if len(unique_labels) > 0: base_label = unique_labels[0]
        elif i == 0: label_for_this_curve = group_label
        if base_label is not None:
            label_for_this_curve = f"{y_col}: {base_label}" if prepend_y_col_to_label else str(base_label)
        ax.plot(curve_data[x_col].values, curve_data[y_col].values, label=label_for_this_curve, **plot_kwargs)
    ax.grid(True)
    return fig, ax

@app.route('/', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        files = request.files.getlist('data_files')
        if not files or files[0].filename == '':
            return render_template('upload.html', error="No file selected. Please choose one or more files.")
        job_id = uuid.uuid4().hex
        job_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
        os.makedirs(job_dir)
        for file in files:
            original_filename = secure_filename(str(file.filename))
            content = file.read().decode('utf-8', errors='ignore')
            block_boundaries = [m.start() for m in re.finditer(r'^\s*Setup\.?Title', content, re.MULTILINE | re.IGNORECASE)]
            if len(block_boundaries) > 1:
                for i, start_pos in enumerate(block_boundaries):
                    end_pos = block_boundaries[i+1] if i + 1 < len(block_boundaries) else len(content)
                    sub_content_block = content[start_pos:end_pos]
                    part_filename = f"{os.path.splitext(original_filename)[0]}_part_{i+1}.csv"
                    with open(os.path.join(job_dir, part_filename), 'w', encoding='utf-8') as f_out:
                        f_out.write(sub_content_block)
            else:
                with open(os.path.join(job_dir, original_filename), 'w', encoding='utf-8') as f_out:
                    f_out.write(content)
        return redirect(url_for('job_page', job_id=job_id))
    return render_template('upload.html')

@app.route('/job/<job_id>', methods=['GET', 'POST'])
def job_page(job_id):
    job_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    if not os.path.isdir(job_dir): abort(404)
    filenames = sorted([f for f in os.listdir(job_dir) if os.path.isfile(os.path.join(job_dir, f))])
    if not filenames: return redirect(url_for('upload_page'))

    context = {"job_id": job_id, "filenames": filenames, "plot_urls": [], "headers": [], "settings": {}, "error": None, "form_data": request.form.to_dict(), "num_plots": request.args.get('num_plots', type=int)}

    try:
        first_file_path = os.path.join(job_dir, filenames[0])
        file_info = parse_file(first_file_path)
        if file_info["error"]: raise ValueError(file_info["error"])
        context["settings"], header_row = file_info["settings"], file_info["header_row"]
        df_template = pd.read_csv(first_file_path, skiprows=header_row, low_memory=False, comment='#')
        context["headers"] = df_template.columns.str.strip().tolist()

        if request.method == 'POST':
            selected_files_to_plot = request.form.getlist('files_to_plot')
            if not selected_files_to_plot: raise ValueError("You must select at least one file.")
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
                if curves_to_plot and len(curves_to_plot) == len(curve_labels): labels_dict = dict(zip(curves_to_plot, curve_labels))
                elif not curves_to_plot and curve_labels: labels_dict = {i: label for i, label in enumerate(curve_labels)}
            prepend_y_col = request.form.get('prepend_y_col') == 'on'
            
            for i in range(1, num_plots + 1):
                numerator, denominator = request.form.get(f'numerator_{i}'), request.form.get(f'denominator_{i}')
                deriv_y, deriv_x = request.form.get(f'deriv_y_{i}'), request.form.get(f'deriv_x_{i}')
                
                y_columns_details = []
                for j in range(1, 6):
                    y_col = request.form.get(f'y_col_{i}_{j}')
                    if y_col:
                        style = request.form.get(f'linestyle_{i}_{j}', 'solid')
                        y_columns_details.append({"name": y_col, "style": style})

                if not any([y_columns_details, (numerator and denominator), (deriv_y and deriv_x)]): continue
                
                fig, ax = plt.subplots(figsize=(12, 8))
                final_ylabel, plot_paths = "", []
                
                if numerator and denominator:
                    final_ylabel = request.form.get(f'ylabel_{i}', '').strip() or f"({numerator}/{denominator})"
                    plot_paths = [{"type": "calc", "y_col": f"({numerator} / {denominator})", "numerator": numerator, "denominator": denominator, "style": "solid"}]
                elif deriv_y and deriv_x:
                    final_ylabel = request.form.get(f'ylabel_{i}', '').strip() or f"d({deriv_y})/d({deriv_x})"
                    plot_paths = [{"type": "deriv", "y_col": f"d({deriv_y})/d({deriv_x})", "deriv_y": deriv_y, "deriv_x": deriv_x, "style": "solid"}]
                elif y_columns_details:
                    final_ylabel = request.form.get(f'ylabel_{i}', '').strip() or ", ".join(d['name'] for d in y_columns_details)
                    plot_paths = [{"type": "direct", "y_col": d['name'], "style": d['style']} for d in y_columns_details]
                
                for path in plot_paths:
                    for file_idx, filename in enumerate(filenames):
                        if filename not in selected_files_to_plot: continue
                        
                        current_file_path = os.path.join(job_dir, filename)
                        file_info_loop = parse_file(current_file_path)
                        header_row_for_this_file = file_info_loop["header_row"]
                        num_curves = request.form.get(f'num_curves_{file_idx}', type=int)
                        if not num_curves: raise ValueError(f"'Total # of Curves' for {filename} is required.")
                        
                        df = pd.read_csv(current_file_path, skiprows=header_row_for_this_file, low_memory=False, comment='#')
                        df.columns = df.columns.str.strip()
                        label_col_data = df[label_source_col].copy() if label_source_col and label_source_col in df else None
                        
                        y_col_to_plot = path["y_col"]
                        if path.get("type") == "calc":
                            num, den = path["numerator"], path["denominator"]
                            if num not in df.columns or den not in df.columns: raise ValueError(f"'{num}' or '{den}' not in {filename}")
                            df[y_col_to_plot] = pd.to_numeric(df[num], errors='coerce') / pd.to_numeric(df[den], errors='coerce')
                        elif path.get("type") == "deriv":
                            dy_col, dx_col = path["deriv_y"], path["deriv_x"]
                            deriv_period = int(request.form.get(f'deriv_period_{i}', 1))
                            if deriv_period < 1: deriv_period = 1
                            if dy_col not in df.columns or dx_col not in df.columns: raise ValueError(f"'{dy_col}' or '{dx_col}' not in {filename}")
                            points_per_curve = len(df) // num_curves
                            curve_ids = np.repeat(np.arange(num_curves), points_per_curve)
                            if len(curve_ids) != len(df): curve_ids = np.resize(curve_ids, len(df))
                            df['curve_id'] = curve_ids
                            numeric_y = pd.to_numeric(df[dy_col], errors='coerce')
                            numeric_x = pd.to_numeric(df[dx_col], errors='coerce')
                            y_diff = numeric_y.groupby(df['curve_id']).diff(periods=deriv_period)
                            x_diff = numeric_x.groupby(df['curve_id']).diff(periods=deriv_period)
                            df[y_col_to_plot] = y_diff / x_diff
                        
                        df_numeric = df.apply(pd.to_numeric, errors='coerce')
                        if label_col_data is not None: df_numeric[label_source_col] = label_col_data
                        
                        if scale_col := request.form.get(f'scale_col_{i}', '').strip():
                            if scale_factor_str := request.form.get(f'scale_factor_{i}'):
                                if scale_col in df_numeric.columns: df_numeric[scale_col] *= float(scale_factor_str)
                        
                        file_alias = request.form.get(f'file_alias_{file_idx}', '').strip() or os.path.splitext(filename)[0]
                        group_label = f"{y_col_to_plot} ({file_alias})"
                        
                        plot_segmented_curves(
                            df=df_numeric, x_col=x_col, y_col=y_col_to_plot, num_curves=num_curves, ax=ax,
                            curves_to_plot=curves_to_plot, labels=labels_dict, label_source_col=label_source_col, 
                            label=group_label, lw=2, linestyle=path['style'], prepend_y_col_to_label=prepend_y_col,
                            file_display_name=file_alias, num_files_plotting=len(selected_files_to_plot)
                        )
                
                final_xlabel = str(request.form.get('xlabel', '') or x_col); final_ylabel = str(final_ylabel)
                auto_title = f'Plot of {final_ylabel} vs {final_xlabel}'
                final_title = str(request.form.get(f'title_{i}', '').strip() or auto_title)
                ax.set_title(final_title); ax.set_xlabel(final_xlabel); ax.set_ylabel(final_ylabel)
                if request.form.get(f'legend_{i}') == 'on': ax.legend()
                if request.form.get('x_log') == 'on': ax.set_xscale('log')
                if request.form.get(f'y_log_{i}') == 'on': ax.set_yscale('log')
                def to_float(val): return float(val) if val else None
                ax.set_xlim(left=to_float(request.form.get('x_min')), right=to_float(request.form.get('x_max')))
                ax.set_ylim(bottom=to_float(request.form.get(f'y_min_{i}')), top=to_float(request.form.get(f'y_max_{i}')))
                
                img = io.BytesIO()
                fig.savefig(img, format='png', bbox_inches='tight', dpi=300)
                img.seek(0)
                context["plot_urls"].append(base64.b64encode(img.getvalue()).decode('utf8'))
                plt.close(fig)
    except Exception as e:
        context["error"] = f"An error occurred: {e}"
        traceback.print_exc()
    return render_template('plot.html', **context)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')