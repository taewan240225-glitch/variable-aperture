# Variable_Aperture

Variable_Aperture is a design and analysis tool for variable iris apertures.

## Files

- `app.py`: Streamlit web app for aperture geometry, blade visualization, gap analysis, and Excel export.
- `index.html`: Browser-runnable stlite version of the Streamlit app.
- `iris_design.py`: PySide6 desktop GUI with the fuller design, measurement, roundness, and export workflow.
- `requirements.txt`: Python dependencies for the Streamlit and desktop GUI apps.

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

After installing the requirements, run the desktop GUI with:

```powershell
python iris_design.py
```
