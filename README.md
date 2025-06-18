# A small plotter for data exported from B1500

By **Bao TON - APC**, assisted by **Gemini**

A powerful, browser-based data visualization tool built with Flask. This application allows users to upload one or more scientific data files 
(CSV), perform on-the-fly calculations, and generate multiple, highly customized plots for comparative analysis.

## Setup and Installation
To run this project on your local machine, follow these steps.

Prerequisites:

* Python 3.6+
* pip (Python package installer)

(at somepoint you may need to use the virtual enviroment: )

```
python -m venv path/to/vent

source path/to/venv/bin/activate 

```

1. Install the required package 

```

pip install -r requirements.txt

```

2. To run the file 

```

python app.py

```

3. To show the web (if it doesn't pop up)

navigate to http://127.0.0.1:5000


## Project Structure
```

your-repository-name/
├── uploads/
├── templates/
│   ├── upload.html
│   └── plot.html
├── app.py
├── requirements.txt
└── README.md

```
