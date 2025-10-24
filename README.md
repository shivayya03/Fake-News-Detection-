📰 Fake News Detection with Machine Learning
Combatting misinformation using NLP and supervised learning — a real-world ML project built for impact and clarity.
📌 Overview
This project classifies news articles as real or fake using machine learning. It demonstrates how data science can be applied to social challenges, with a focus on interpretability, performance, and educational value.
📊 Dataset
- Source: Kaggle Fake News Dataset
- Size: 20,000+ articles
- Features: Title, text, subject, date
- Target: Binary label — REAL or FAKE

🧠 Model Pipeline
- Preprocessing:
- Tokenization
- Stopword removal
- TF-IDF vectorization
- Algorithms Used:
- Logistic Regression
- Passive Aggressive Classifier
- Performance:
- Accuracy: 92%
- Precision: 91%
- Recall: 93%
- F1 Score: 92%
📁 Project Structure
├── data/                  # Raw and cleaned datasets
├── notebooks/             # EDA and model development
├── models/                # Saved model files
├── utils/                 # Preprocessing functions
├── app.py                 # Streamlit app for demo
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
🧪 How to Run
# Clone the repository
git clone https://github.com/shivayya03/Fake-News-Detection.git

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
📈 Visuals
- Confusion matrix
- Classification report
- Word clouds for fake vs real news
- Feature importance plots
🎯 Key Learnings
- Applied NLP to real-world classification
- Compared models using multiple metrics
- Built an interactive dashboard for public engagement
🔮 Future Scope
- Integrate deep learning models (e.g., LSTM, BERT)
- Expand dataset with multilingual sources
- Deploy as a browser extension for real-time detection
🙋‍♂️ About Me
I’m Shivayya, a B.Tech student passionate about machine learning, data visualization, and educational outreach. This project reflects my drive to build impactful tools and share technical knowledge in engaging ways.

