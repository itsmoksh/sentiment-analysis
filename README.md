# Project Report 

---
## Sentiment Analysis
This is a **Sentiment Analysis Web App** built using **Streamlit** and **Hugging Face Transformers**.  
The model is based on `bert-base-uncased` and classifies reviews into **Positive** or **Negative** sentiment.

---
## Model Details 

1. Model was trained on Kaggle IMDB Dataset of 50K movies
   - Contains 25K Positive and 25K Negative reviews
2. Used Pre-trained Bert Model
   - Froze weights for better results(Transfer Learning)
   - Convert text to token's id and their corresponding contextual embedding
3. The accuracy on validation set was around 85%

---
## Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python, Pytorch, Bert
- **Deployment**: Streamlit Cloud

---
## Project Structure
``` text
sentiment-analysis/
│
├── screenshots/                
│   ├── img1      #Positive Review
│   ├── img2      #Negative Review
│
├── streamlit/        
│   ├──app.py         #Stremlit UI
│   ├──prediction.py  #Backend
│    
├── requirements.txt            # Python dependencies
├── LICENSE                     # Apache 2.0 license
└── README.md                   # Project documentation
```


---
## How to run locally

1. Clone the repo:
``` bash
git clone https://github.com/itsmoksh/sentiment-analysis.git
cd sentiment-analysis
```
2. Install Dependencies:
``` bash
pip install -r requirements.txt
```
3. Run the app:
``` bash
streamlit run app.py
```
---

## Live App

Check out the **live demo**:  
[Streamlit Cloud Link](#) *(https://moksh-sentiment-analysis.streamlit.app/)*

##  Author

**Moksh Jain**  
Aspiring Data Scientist | Python & ML Enthusiast  
[LinkedIn](https://www.linkedin.com/in/itsmoksh/) • [GitHub](https://github.com/itsmoksh)
  




