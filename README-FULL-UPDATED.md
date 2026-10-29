# Forma — Full Updated Project

This package combines the existing Data Studio pages with the Feature Intelligence backend and UI.

## Studio navigation
- The sidebar studio selector switches between **Data Studio** and **ML Studio**.
- Only the selected studio’s navigation group is shown at a time.
- Data Studio retains Data Profiling, Data Cleaning, Feature Engineering, and Data Explorer.
- ML Studio includes Pipeline Studio, Experiment Lab, Explainability, Feature Intelligence, and Model Registry.

## Run backend (Windows PowerShell)
```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Use the Python virtual environment already set up for this project. Install `requirements.txt` in that environment if needed.

## Run frontend
```powershell
cd frontend
npm install
npm run dev
```

## Feature Intelligence
Upload a CSV using the existing New ML project flow. Then open ML Studio → Feature Intelligence. The page uses the uploaded dataset ID and backend endpoint for feature importance, feature selection, and PCA. If you open the page without a dataset, upload one first.

## Important
- The full project is supplied, not only a sidebar patch.
- Existing Data Profiling and Data Cleaning component files are retained.
- The project has not been tested in your local Windows environment; run the commands above and share any build/runtime error for a targeted fix.
