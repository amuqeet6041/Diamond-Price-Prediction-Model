# Diamond Price Prediction

Predicts the price (USD) of a diamond from its physical and quality specifications using **K-Fold Cross Validation** and 5 regression algorithms.

**Live App:** [Diamond Price Predictor](<https://diamond-price-prediction-model-the-muqeet.streamlit.app/>)

---

## Files in This Folder

| File | Description |
|---|---|
| `diamond_price_kfold_full_dataset_10fold.ipynb` | Main Jupyter notebook — runs on Google Colab |
| `Diamonds Prices2022.csv` | Training dataset (53,940 diamonds, CC BY 4.0) |
| `model.pkl` | Best trained pipeline (created when notebook runs) |
| `app.py` | Streamlit web app |
| `requirements.txt` | Pinned Python dependencies |

---

## Run the Notebook (Google Colab)

1. Go to **https://colab.research.google.com** and sign in.
2. Click **File → Upload notebook** and choose `diamond_price_kfold_full_dataset_10fold.ipynb`.
3. Click **Runtime → Run all**.
4. When **Step 2.1** runs, click **"Choose Files"** and upload `Diamonds Prices2022.csv`.
5. After the notebook finishes, download `model.pkl` from the Colab **Files** panel (set `DOWNLOAD_MODEL = True` in Step 10.3 or use the folder icon on the left).

---

## Deploy on Streamlit Community Cloud

> **Prerequisite:** `model.pkl` must be committed to your GitHub repo. The notebook creates it; make sure to add and push it.

1. **Push this folder to GitHub**
   ```bash
   git add "2 - Diamond Prices Prediction Model/"
   git commit -m "Add diamond price prediction model and app"
   git push
   ```

2. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

3. Click **New app** and fill in:
   - **Repository:** `your-username/your-repo`
   - **Branch:** `main`
   - **Main file path:** `2 - Diamond Prices Prediction Model/app.py`

4. Click **Deploy**. Streamlit will install `requirements.txt` automatically.

5. Once deployed, **copy the live URL** and paste it into:
   - The first cell of the notebook (replace `<PASTE_LINK_HERE>`)
   - The `README.md` above (replace `<PASTE_LINK_HERE>`)

---

## Dataset Credit

*Diamonds Prices 2022* — Kaggle, **CC BY 4.0**.  
Source: <https://www.kaggle.com/datasets/nancyalaswad90/diamonds-prices>
