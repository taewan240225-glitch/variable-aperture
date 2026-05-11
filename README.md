# Variable_Aperture

Variable_Aperture is a design and analysis tool for variable iris apertures.

## Files

- `app.py`: Streamlit web app for aperture geometry, blade visualization, gap analysis, and Excel export.
- `index.html`: Browser-runnable stlite version of the Streamlit app.
- `iris_design.py`: PySide6 desktop GUI with the fuller design, measurement, roundness, and export workflow.
- `requirements.txt`: Python dependencies for the Streamlit app.

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

For the desktop GUI, install the additional GUI dependencies first:

```powershell
pip install PySide6 pillow
python iris_design.py
```
